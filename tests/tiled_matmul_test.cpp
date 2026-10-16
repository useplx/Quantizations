#include "Q4K/q4k.hpp"
#include "Q8/q8.hpp"
#include "syn/compiler.hpp"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/Verifier.h"
#include "mln/sdk/functional/matmul.hpp"
#include "mln/sdk/functional/transpose.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>

namespace {
    class Tiling final : public Mln::Sdk::Operation::Compatibility {
        public:
            explicit Tiling(int64_t rows, int64_t contraction, bool unroll, bool linearize,
                            bool fuse_producer)
                : rows_(rows), contraction_(contraction), unroll_(unroll), linearize_(linearize),
                  fuse_producer_(fuse_producer) {}

            bool applies(const Context &context) const override {
                return context.operation.kind == Mln::Sdk::Functional::Kind::MatMul;
            }

            int64_t tile(const Context &, int64_t axis, int64_t extent) const override {
                return axis == -1 && rows_ > 0 ? std::min(rows_, extent) : extent;
            }

            std::optional<bool> linearize_reduction(const Context &) const override {
                return linearize_;
            }
            std::optional<bool> fuse_producer(const Context &) const override {
                return fuse_producer_;
            }
            bool unroll(const Context &) const override { return unroll_; }
            std::optional<int64_t> contraction_tile(const Context &) const override {
                return linearize_ ? std::optional<int64_t>{contraction_} : std::nullopt;
            }

        private:
            int64_t rows_;
            int64_t contraction_;
            bool unroll_;
            bool linearize_;
            bool fuse_producer_;
    };

    class Projection final : public Mln::Sdk::Module {
        public:
            explicit Projection(int64_t rows, std::string module_name = "projection") {
                name = std::move(module_name);
                parameters.insert("weight", {{}, {rows, 2048}});
            }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                auto weight = operations.insert<Mln::Sdk::Functional::Transpose>(
                    parameters.at("weight"), std::vector<int64_t>{1, 0});
                return operations.insert<Mln::Sdk::Functional::MatMul>(input, std::move(weight));
            }
    };

    void check(int64_t rows, std::vector<int64_t> shape, const std::string &output,
               std::string_view target = "local", int32_t subgroup = 0, int64_t tile_rows = 64,
               std::string_view abi = "descriptors", std::string_view target_features = {},
               bool unroll = false, bool q4k = false, int64_t contraction_tile = 64,
               std::optional<bool> linearize = std::nullopt, bool fuse_producer = true,
               std::string module_name = "projection",
               std::optional<std::pair<std::string, std::string>> diagnostic_flag = std::nullopt) {
        Tiling tiling(tile_rows, contraction_tile, unroll, linearize.value_or(target != "local"),
                      fuse_producer);
        Mln::Sdk::Codec::Format format = q4k ? Mln::Sdk::Codec::Format(Quantizations::Q4K{})
                                             : Mln::Sdk::Codec::Format(Quantizations::Q8{});
        Syn::Compiler::Manifest manifest{
            .policies = {{"projection",
                          Syn::Compiler::Police{
                              .schemes = {Mln::Sdk::Codec::Scheme{
                                  .scope = Mln::Sdk::Codec::Scope::Constant,
                                  .lifetime = Mln::Sdk::Codec::Lifetime::Immutable,
                                  .update = Mln::Sdk::Codec::Update::None,
                                  .format = format,
                              }},
                              .dtype = Mln::DType::Float16,
                              .backend =
                                  Mln::Sdk::Backend{
                                      .device = "local",
                                      .flags = {{"iree-hal-local-target-device-backends",
                                                 "llvm-cpu"}},
                                  },
                          }}},
            .formats = {format},
            .operation_compatibilities = {&tiling},
        };
        if (target != "local") {
            manifest.policies.front().second.backend = Mln::Sdk::Backend{
                .device = "vulkan",
                .flags = {{"iree-vulkan-target", std::string(target)},
                          {"iree-vulkan-dispatch-abi", std::string(abi)},
                          {"iree-hal-dump-executable-sources-to", output + "-executables"}},
            };
            if (!target_features.empty())
                manifest.policies.front().second.backend->flags.emplace_back(
                    "iree-vulkan-target-features", target_features);
            if (diagnostic_flag)
                manifest.policies.front().second.backend->flags.push_back(*diagnostic_flag);
        }
        Projection projection(rows, std::move(module_name));
        Syn::Compiler::Compiler compiler(manifest, "tiling");
        if (subgroup != 0) {
            namespace IREE = mlir::iree_compiler::IREE;
            auto *context = compiler.context.get();
            context->getOrLoadDialect<IREE::GPU::IREEGPUDialect>();
            auto gpu = IREE::GPU::getVulkanTargetDetails(target, context);
            if (!gpu || subgroup <= 0)
                throw std::runtime_error("invalid diagnostic GPU target");
            auto wgp = gpu.getWgp();
            llvm::SmallVector<mlir::Attribute> attributes;
            llvm::SmallVector<mlir::Type> types;
            wgp.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    attributes.push_back(attr == wgp.getSubgroupSizeChoices()
                                             ? mlir::DenseI32ArrayAttr::get(context, {subgroup})
                                             : attr);
                },
                [&](mlir::Type type) { types.push_back(type); });
            wgp = mlir::cast<IREE::GPU::TargetWgpAttr>(
                wgp.replaceImmediateSubElements(attributes, types));
            std::string features =
                target_features.empty() ? gpu.getFeatures().str() : std::string(target_features);
            if (abi == "bda")
                features += ",cap:Int64,cap:PhysicalStorageBufferAddresses,ext:SPV_KHR_physical_"
                            "storage_buffer";
            gpu = IREE::GPU::TargetAttr::get(context, gpu.getArch(), features, wgp, gpu.getChip());
            mlir::Builder builder(context);
            auto executable = IREE::HAL::ExecutableTargetAttr::get(
                context, builder.getStringAttr("vulkan-spirv"),
                builder.getStringAttr(abi == "bda" ? "vulkan-spirv-bda-v1" : "vulkan-spirv-fb"),
                builder.getDictionaryAttr({builder.getNamedAttr("iree_codegen.target_info", gpu)}));
            auto device = IREE::HAL::DeviceTargetAttr::get(
                context, builder.getStringAttr("vulkan"), builder.getDictionaryAttr({}),
                llvm::ArrayRef<IREE::HAL::ExecutableTargetAttr>{executable});
            std::string spec;
            llvm::raw_string_ostream target_stream(spec);
            target_stream << device;
            manifest.policies.front().second.backend->flags.emplace_back("iree-hal-target-device",
                                                                         spec);
            if (!output.empty())
                std::ofstream(output + ".target") << spec;
        }
        Mln::Sdk::Tensor input(std::move(shape));
        const auto start = std::chrono::steady_clock::now();
        compiler.compile_as_impl("projection", "projection", true, projection, input);
        if (mlir::failed(mlir::verify(*compiler.module)))
            throw std::runtime_error("tiled projection has invalid SSA or types");
        std::size_t loops = 0;
        std::size_t contractions = 0;
        compiler.module->walk([&](mlir::Operation *operation) {
            loops += mlir::isa<mlir::stablehlo::WhileOp>(operation);
            contractions += mlir::isa<mlir::linalg::GenericOp>(operation);
        });
        const int64_t tile = tile_rows > 0 ? tile_rows : rows;
        const bool looped = rows > tile && !unroll;
        if (loops != static_cast<std::size_t>(looped) || contractions == 0)
            throw std::runtime_error(std::format(
                "output tiling expected loops={} with contractions, got loops={} contractions={}",
                looped, loops, contractions));
        std::string graph;
        llvm::raw_string_ostream stream(graph);
        compiler.module->print(stream);
        if (looped && graph.size() > 30000)
            throw std::runtime_error("output tiling IR grows with output width");
        std::println(
            "rows={} rank={} loops={} contractions={} IR={} bytes time={} ms", rows,
            input.shape.size(), loops, contractions, graph.size(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
        if (!output.empty()) {
            std::ofstream(output + ".mlir") << graph;
            compiler.write_vmfb(output + ".vmfb", output + ".devices", output + ".weights");
            if (subgroup != 0) {
                std::ifstream source(output +
                                     "-executables/module_tiling_projection_dispatch_0.mlir");
                const std::string executable{std::istreambuf_iterator<char>(source), {}};
                if (executable.find(std::format("subgroup_size_choices = [{}]", subgroup)) ==
                    std::string::npos)
                    throw std::runtime_error("explicit target was replaced by the backend alias");
            }
        }
    }
}

int main(int argc, char **argv) try {
    for (const int64_t rows : {64, 128, 130, 6144, 248320}) {
        check(rows, {2048}, {});
        check(rows, {8, 2048}, {});
    }
    check(128, {2048}, {}, "local", 0, 64, "descriptors", {}, true);
    check(128, {8, 2048}, {}, "local", 0, 64, "descriptors", {}, true);
    if (argc >= 2) {
        const int64_t batch = argc >= 6 ? std::stoll(argv[5]) : 2;
        check(argc >= 4 ? std::stoll(argv[3]) : 128,
              batch == 0 ? std::vector<int64_t>{2048} : std::vector<int64_t>{batch, 2048}, argv[1],
              argc >= 3 ? argv[2] : "local", argc >= 5 ? std::stoi(argv[4]) : 0,
              argc >= 7 ? std::stoll(argv[6]) : 64, argc >= 8 ? argv[7] : "descriptors",
              argc >= 9 ? argv[8] : "", argc >= 10 && std::string_view(argv[9]) == "unroll",
              argc >= 11 && std::string_view(argv[10]) == "q4k",
              argc >= 12 ? std::stoll(argv[11]) : 64,
              argc >= 13 ? std::optional<bool>{std::string_view(argv[12]) == "linear"}
                         : std::nullopt,
              argc < 14 || std::string_view(argv[13]) != "resolve",
              argc >= 15 ? argv[14] : "projection",
              argc >= 17 ? std::optional<std::pair<std::string, std::string>>{{argv[15], argv[16]}}
                         : std::nullopt);
    }
    return 0;
} catch (const std::exception &error) {
    std::println(stderr, "{}", error.what());
    return 1;
}
