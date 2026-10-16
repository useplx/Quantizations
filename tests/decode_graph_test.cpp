#include "Q4K/q4k.hpp"
#include "Q5K/q5k.hpp"
#include "Q6K/q6k.hpp"
#include "Q8/q8.hpp"

#include <mln/sdk/operations.hpp>

#include <cstdint>
#include <cstdio>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
    using Kind = Mln::Sdk::Functional::Kind;

    std::string_view name_of(Kind kind) {
        switch (kind) {
        case Kind::Concatenate:
            return "Concatenate";
        case Kind::Transpose:
            return "Transpose";
        default:
            return "?";
        }
    }

    template <typename Codec>
    void require_composable_graph(std::string_view name, Codec codec, std::vector<int64_t> shape) {
        Mln::Sdk::Operations operations;
        const Mln::Sdk::Weight weight{std::string("model.w"), shape};
        const Mln::Sdk::Tensor decoded = codec.decode(operations, weight, Mln::DType::Float16);

        if (decoded.shape != shape)
            throw std::runtime_error(std::string(name) +
                                     ": decode returned a shape different from the weight");

        std::size_t nodes = 0;
        for (const auto &operation : operations) {
            ++nodes;
            if (operation->kind == Kind::Concatenate || operation->kind == Kind::Transpose)
                throw std::runtime_error(std::string(name) + ": decode emitiu " +
                                         std::string(name_of(operation->kind)) +
                                         " -- fragmenta o caminho ate o consumidor");
        }
        std::println("{} nodes={} ok", name, nodes);
    }
}

int main() {
    const std::vector<int64_t> shape{64, 256};
    try {
        require_composable_graph("Q4K", Quantizations::Q4K{}, shape);
        require_composable_graph("Q5K", Quantizations::Q5K{}, shape);
        require_composable_graph("Q6K", Quantizations::Q6K{}, shape);
        require_composable_graph("Q8", Quantizations::Q8{}, shape);
    } catch (const std::exception &error) {
        std::println(stderr, "failed: {}", error.what());
        return 1;
    }
    return 0;
}
