#include "Q8/q8.hpp"
#include "quant/compatibility.hpp"

#include <array>
#include <stdexcept>

int main() {
    using namespace Mln::Sdk;
    using Kind = Functional::Kind;
    Quantizations::Q8 codec;
    Codec::Scheme scheme;
    Weight weight{"model.weight", {256, 256}};
    Functional::MatMul matmul{Tensor{{256}}, Tensor{{256, 256}}};
    Functional::Transpose transpose{Tensor{{256, 256}}, {1, 0}};
    Functional::Add add{Tensor{{256}}, Tensor{{256}}};
    Quantizations::Compatibility compatibility{"Q8_GROUPED", {Kind::MatMul}};
    const auto check = [&](std::span<const Operation *const> consumers, std::string_view backend,
                           bool expected) {
        const Syn::Compiler::Codec::Compatibility::Context context{
            .codec = codec,
            .scheme = scheme,
            .weight = weight,
            .dtype = Mln::DType::Float16,
            .parameter_path = weight.name,
            .backend = backend,
            .consumers = consumers,
        };
        if (compatibility.applies(context) != expected)
            throw std::runtime_error("incorrect codec consumer selection");
    };

    const std::array<const Operation *, 3> contraction{&add, &matmul, &transpose};
    check(contraction, "local", true);
    check(contraction, "cuda", true);
    check(contraction, "unknown", false);

    const std::array<const Operation *, 3> arithmetic{&matmul, &add, &transpose};
    check(arithmetic, "cuda", false);
    const std::array<const Operation *, 1> layout{&transpose};
    check(layout, "local", false);
    check({}, "local", false);
    weight.shape = {256, 31};
    check(contraction, "cuda", false);
}
