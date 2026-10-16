#pragma once

#include <syn/codec_compatibility.hpp>

#include <algorithm>
#include <initializer_list>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace Quantizations {
    class Compatibility final : public Syn::Compiler::Codec::Compatibility {
        public:
            Compatibility(std::string storage,
                          std::initializer_list<Mln::Sdk::Functional::Kind> consumers)
                : storage_(std::move(storage)), consumers_(consumers) {}

            bool applies(const Context &context) const override {
                if (context.codec.storage_name() != storage_ || context.consumers.empty())
                    return false;

                if (!supports(context.backend) ||
                    !context.codec.supports(context.weight, context.scheme))
                    return false;

                for (const auto *operation : context.consumers | std::views::reverse) {
                    if (!operation)
                        return false;
                    const auto kind = operation->kind;
                    if (kind == Mln::Sdk::Functional::Kind::Transpose ||
                        kind == Mln::Sdk::Functional::Kind::Reshape)
                        continue;
                    return std::ranges::find(consumers_, kind) != consumers_.end();
                }
                return false;
            }

        private:
            static bool supports(std::string_view backend) {
                return backend == "local" || backend == "cuda" || backend == "vulkan";
            }

            std::string storage_;
            std::vector<Mln::Sdk::Functional::Kind> consumers_;
    };
}
