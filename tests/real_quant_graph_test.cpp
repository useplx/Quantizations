#include "Q4K/q4k.hpp"
#include "Q5K/q5k.hpp"
#include "Q6K/q6k.hpp"
#include "Q8/q8.hpp"

#include "syn/compiler.hpp"

#include "iree/compiler/embedding_api.h"
#include "mln/sdk/functional.hpp"
#include "mln/sdk/module.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using MQ = Mln::Sdk::Codec;

    template <typename Format> MQ::Scheme constant(Format format) {
        return MQ::Scheme{
            .scope = MQ::Scope::Constant,
            .lifetime = MQ::Lifetime::Immutable,
            .update = MQ::Update::None,
            .format = std::move(format),
        };
    }

    class VulkanReduction final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return context.target && context.target->device == "vulkan" &&
                       context.operation.kind == Mln::Sdk::Functional::Kind::MatMul;
            }

            std::optional<bool> linearize_reduction(const Context &) const override { return true; }
    };

    class Graph final : public Mln::Sdk::Module {
        public:
            Graph() {
                name = "model";
                parameters.insert("q4", {{}, {256, 256}});
                parameters.insert("q5", {{}, {256, 256}});
                parameters.insert("q6", {{}, {256, 256}});
                parameters.insert("q8", {{}, {256, 256}});
                parameters.insert("emb", {{}, {256, 256}});
            }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                for (const char *parameter : {"q4", "q5", "q6", "q8"}) {
                    Mln::Sdk::Tensor weight = parameters.at(parameter);
                    weight = operations.insert<Mln::Sdk::Functional::Transpose>(
                        std::move(weight), std::vector<int64_t>{1, 0});
                    input = operations.insert<Mln::Sdk::Functional::MatMul>(std::move(input),
                                                                            std::move(weight));
                }

                Mln::Sdk::Tensor index = operations.insert<Mln::Sdk::Functional::Const>(
                    Mln::Sdk::Weight{"", {1}, 0.0}, false, Mln::DType::Int32);
                Mln::Sdk::Tensor row = operations.insert<Mln::Sdk::Functional::Gather>(
                    parameters.at("emb"), std::move(index), 0);
                row = operations.insert<Mln::Sdk::Functional::Reshape>(std::move(row),
                                                                       std::vector<int64_t>{256});
                return operations.insert<Mln::Sdk::Functional::Multiply>(std::move(input),
                                                                         std::move(row));
            }
    };

    class Q8MatMul final : public Mln::Sdk::Module {
        public:
            Q8MatMul() {
                name = "q8only";
                parameters.insert("weight", {{}, {6144, 2048}});
            }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                Mln::Sdk::Tensor weight = operations.insert<Mln::Sdk::Functional::Transpose>(
                    parameters.at("weight"), std::vector<int64_t>{1, 0});
                return operations.insert<Mln::Sdk::Functional::MatMul>(std::move(input),
                                                                       std::move(weight));
            }
    };

    class Q8Gather final : public Mln::Sdk::Module {
        public:
            Q8Gather() {
                name = "q8gather";
                parameters.insert("weight", {{}, {6144, 2048}});
            }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor indices) {
                return operations.insert<Mln::Sdk::Functional::Gather>(parameters.at("weight"),
                                                                       std::move(indices), 0);
            }
    };

    void require_contains(const std::string &graph, std::string_view token) {
        if (graph.find(token) == std::string::npos)
            throw std::runtime_error("quantization graph does not contain '" + std::string(token) +
                                     "'");
    }

    std::size_t count_occurrences(const std::string &text, std::string_view token) {
        std::size_t count = 0;
        for (std::size_t at = 0; (at = text.find(token, at)) != std::string::npos;
             at += token.size())
            ++count;
        return count;
    }

    struct Dispatch {
            std::string name;
            std::string body;

            bool gathers() const {
                return body.find("tensor.extract") != std::string::npos ||
                       name.find("gather") != std::string::npos;
            }

            bool contracts() const {
                return name.find("matmul") != std::string::npos ||
                       name.find("vecmat") != std::string::npos ||
                       name.find("matvec") != std::string::npos;
            }

            bool unpacks() const {
                return body.find("arith.andi") != std::string::npos &&
                       (body.find("arith.shrui") != std::string::npos ||
                        body.find("arith.shli") != std::string::npos);
            }

            bool dequantizes() const {
                return body.find("arith.uitofp") != std::string::npos &&
                       body.find("arith.subf") != std::string::npos &&
                       body.find("arith.mulf") != std::string::npos;
            }

            bool materializes_decoded_weight() const {
                return body.find("writeonly:tensor<256x256xf16>") != std::string::npos ||
                       body.find("writeonly:tensor<8x32x256xf16>") != std::string::npos ||
                       body.find("writeonly:tensor<4x2x32x256xf16>") != std::string::npos ||
                       body.find("writeonly:tensor<2x2x2x2x16x256xf16>") != std::string::npos ||
                       body.find("writeonly:tensor<2048x32xf16>") != std::string::npos;
            }
    };

    std::vector<Dispatch> read_dispatches(const std::filesystem::path &directory) {
        std::vector<Dispatch> dispatches;
        for (const auto &entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().extension() != ".mlir")
                continue;
            std::ifstream file(entry.path());
            std::ostringstream buffer;
            buffer << file.rdbuf();
            const std::string body = buffer.str();

            const std::string marker = "hal.executable.export public @";
            const std::size_t begin = body.find(marker);
            if (begin == std::string::npos)
                continue;
            const std::size_t from = begin + marker.size();
            const std::size_t end = body.find(' ', from);
            dispatches.push_back(Dispatch{body.substr(from, end - from), body});
        }
        return dispatches;
    }

    void report_decode_dispatches(const std::filesystem::path &directory) {
        constexpr std::size_t dispatch_ceiling = 14;
        constexpr std::size_t copy_ceiling = 0;

        const std::vector<Dispatch> dispatches = read_dispatches(directory);
        if (dispatches.empty())
            throw std::runtime_error("no executable was dumped; measurement did not run");

        std::size_t contractions = 0;
        std::size_t fused = 0;
        std::size_t copies = 0;
        std::size_t canonical = 0;
        std::size_t gathers = 0;
        std::size_t gathers_fused = 0;
        std::size_t decoded_weight_materializations = 0;
        for (const Dispatch &dispatch : dispatches) {
            if (dispatch.materializes_decoded_weight())
                ++decoded_weight_materializations;
            if (dispatch.gathers()) {
                ++gathers;
                if (dispatch.dequantizes())
                    ++gathers_fused;
            }
            if (dispatch.name.find("slow_memcpy") != std::string::npos)
                ++copies;
            if (dispatch.dequantizes())
                ++canonical;
            if (!dispatch.contracts())
                continue;
            ++contractions;
            if (dispatch.unpacks() || dispatch.dequantizes())
                ++fused;
        }

        std::println("dispatches: {} | contracoes: {} | com decode fundido: {} | "
                     "dequant canonico: {} | gathers: {} | gathers com decode: {} | "
                     "materialized decoded weights: {} | slow_memcpy: {}",
                     dispatches.size(), contractions, fused, canonical, gathers, gathers_fused,
                     decoded_weight_materializations, copies);

        if (contractions == 0)
            throw std::runtime_error("no contraction dispatch in the dump; the test is not "
                                     "medindo o que pensa medir");

        if (canonical < 2)
            throw std::runtime_error("apenas " + std::to_string(canonical) +
                                     " canonical decodes (expected at least 2: q4k and q8)");

        if (copies > copy_ceiling)
            throw std::runtime_error(std::to_string(copies) + " dispatches slow_memcpy, teto e " +
                                     std::to_string(copy_ceiling) +
                                     " -- concat de volta no caminho de dados?");

        if (dispatches.size() > dispatch_ceiling)
            throw std::runtime_error(std::to_string(dispatches.size()) + " dispatches, teto e " +
                                     std::to_string(dispatch_ceiling) +
                                     " -- o decode voltou a fragmentar");

        if (fused != contractions)
            throw std::runtime_error(std::to_string(contractions - fused) + " de " +
                                     std::to_string(contractions) +
                                     " contractions still receive the materialized weight");
        if (decoded_weight_materializations != 0)
            throw std::runtime_error(std::to_string(decoded_weight_materializations) +
                                     " complete decoded weights were materialized");
    }
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 8)
        return 2;

    ireeCompilerGlobalInitialize();
    const char *global_flags[] = {
        "mln_real_quant_graph_test",
        "--iree-global-opt-enable-quantized-matmul-reassociation",
    };
    ireeCompilerSetupGlobalCL(static_cast<int>(sizeof(global_flags) / sizeof(global_flags[0])),
                              global_flags, "myelin", false);

    const std::filesystem::path dump_directory = argv[5];
    std::filesystem::remove_all(dump_directory);
    std::filesystem::create_directories(dump_directory);

    VulkanReduction vulkan_reduction;
    Syn::Compiler::Manifest manifest{
        .policies =
            {
                {"model",
                 Syn::Compiler::Police{
                     .schemes = {constant(Quantizations::Q4K{})},
                     .dtype = Mln::DType::Float16,
                     .backend =
                         Mln::Sdk::Backend{
                             .device = "local",
                             .flags =
                                 {
                                     {"iree-hal-local-target-device-backends", "llvm-cpu"},
                                     {"iree-dispatch-creation-enable-aggressive-"
                                      "fusion",
                                      "true"},
                                     {"iree-hal-dump-executable-sources-to",
                                      dump_directory.string()},
                                 },
                         },
                 }},
                {"model.q4",
                 Syn::Compiler::Police{
                     .schemes = {constant(Quantizations::Q4K{})},
                 }},
                {"model.q5",
                 Syn::Compiler::Police{
                     .schemes = {constant(Quantizations::Q5K{})},
                 }},
                {"model.q6",
                 Syn::Compiler::Police{
                     .schemes = {constant(Quantizations::Q6K{})},
                 }},
                {"model.q8",
                 Syn::Compiler::Police{
                     .schemes = {constant(Quantizations::Q8{})},
                 }},
                {"model.emb",
                 Syn::Compiler::Police{
                     .schemes = {constant(Quantizations::Q8{})},
                 }},
                {"q8only",
                 Syn::Compiler::Police{
                     .schemes = {constant(Quantizations::Q8{})},
                     .dtype = Mln::DType::Float16,
                     .backend =
                         Mln::Sdk::Backend{
                             .device =
                                 "local",
                             .flags = {{"iree-hal-local-target-device-backends", "llvm-cpu"}},
                         },
                 }},
                {"q8gather",
                 Syn::Compiler::Police{
                     .schemes = {constant(Quantizations::Q8{})},
                     .dtype = Mln::DType::Float16,
                     .backend =
                         Mln::Sdk::Backend{
                             .device = "local",
                             .flags = {{"iree-hal-local-target-device-backends", "llvm-cpu"}},
                         },
                 }},
            },
        .formats =
            {
                Quantizations::Q4K{},
                Quantizations::Q5K{},
                Quantizations::Q6K{},
                Quantizations::Q8{},
            },
        .operation_compatibilities = {&vulkan_reduction},
    };

    std::vector<Mln::Weight::Descriptor> weights{
        {"model.q4", "F32", "F32", {256, 256}, 256 * 256 * 4},
        {"model.q5", "F32", "F32", {256, 256}, 256 * 256 * 4},
        {"model.q6", "F32", "F32", {256, 256}, 256 * 256 * 4},
        {"model.q8", "F32", "F32", {256, 256}, 256 * 256 * 4},
        {"model.emb", "F32", "F32", {256, 256}, 256 * 256 * 4},
        {"q8only.weight", "F32", "F32", {6144, 2048}, 6144 * 2048 * 4},
        {"q8gather.weight", "F32", "F32", {6144, 2048}, 6144 * 2048 * 4},
    };

    Graph module;
    Syn::Compiler::Compiler compiler(manifest, "quantizations", weights);
    Mln::Sdk::Tensor input({256});
    compiler.compile(module, input);
    Q8MatMul q8_matmul;
    Mln::Sdk::Tensor q8_input({8, 2048});
    compiler.compile(q8_matmul, q8_input);
    Q8Gather q8_gather;
    Mln::Sdk::Tensor q8_indices({8}, Mln::DType::Int32);
    compiler.compile(q8_gather, q8_indices);
    compiler.write_vmfb(argv[1], argv[2], argv[3]);

    if (argc == 8) {
        Syn::Compiler::Manifest vulkan = manifest;
        for (auto &[path, police] : vulkan.policies) {
            (void)path;
            police.backend = Mln::Sdk::Backend{
                .device = "vulkan",
                .flags = {{"iree-vulkan-target", "adreno-643"}},
            };
        }
        Q8MatMul vulkan_q8_matmul;
        Syn::Compiler::Compiler vulkan_compiler(vulkan, "quantizations", weights);
        Mln::Sdk::Tensor vulkan_input({8, 2048});
        vulkan_compiler.compile(vulkan_q8_matmul, vulkan_input);
        Q8Gather vulkan_q8_gather;
        Mln::Sdk::Tensor vulkan_indices({8}, Mln::DType::Int32);
        vulkan_compiler.compile(vulkan_q8_gather, vulkan_indices);
        vulkan_compiler.write_vmfb(argv[6], argv[7]);
    }

    std::ofstream graph(argv[4]);
    if (!graph)
        throw std::runtime_error("could not write StableHLO");
    graph << compiler.graph;
    graph.close();

    require_contains(compiler.graph, "linalg.generic");
    require_contains(compiler.graph, "stablehlo.multiply");
    require_contains(compiler.graph, "stablehlo.shift_right_logical");
    require_contains(compiler.graph, "stablehlo.and");
    if (count_occurrences(compiler.graph, "arith.addf") < 4)
        throw std::runtime_error("the four codecs did not reach IREE as contractions "
                                 "agrupadas");
    if (compiler.graph.find("myelin.quant") != std::string::npos)
        throw std::runtime_error("decode virou operacao opaca myelin.quant");

    report_decode_dispatches(dump_directory);
    return 0;
}
