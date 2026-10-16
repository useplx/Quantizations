#pragma once

#include "quant/common.hpp"
#include "quant/search.hpp"

#include <algorithm>
#include <cmath>

namespace Quantizations {
    class Q4K final : public Codec {
        private:
            using MQ = Mln::Sdk::Codec;

            static constexpr int64_t GROUP_ELEMENTS = 32;
            static constexpr int64_t BITS = 4;
            static constexpr long MINIMUM = 0;
            static constexpr long MAXIMUM = 15;

            static MQ::Scheme default_scheme() {
                return MQ::Scheme{
                    .scope = MQ::Scope::Constant,
                    .lifetime = MQ::Lifetime::Immutable,
                    .update = MQ::Update::None,
                };
            }

        public:
            Q4K() { name = "Q4K"; }

            std::string storage_name() const override { return name; }

            bool supports(const Weight &weight) const override {
                return supports(weight, default_scheme());
            }

            bool supports(const Weight &weight, const MQ::Scheme &) const override {
                return weight.shape.size() == 2 && weight.shape[0] > 0 && weight.shape[1] > 0 &&
                       weight.shape[1] % GROUP_ELEMENTS == 0;
            }

            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                return decode(ops, weight, default_scheme(), dtype);
            }

            Tensor decode(Operations &ops, const Weight &weight, const MQ::Scheme &scheme,
                          DType dtype) override {
                if (!supports(weight, scheme))
                    throw std::invalid_argument("Q4K: unsupported Scheme or shape");
                return Common::decode_affine_grouped(ops, weight, GROUP_ELEMENTS, BITS, dtype);
            }

            std::vector<Mln::Sdk::Codec::Parameter::Component>
            encode(std::span<const float> values, const Weight &weight,
                   const MQ::Scheme &scheme) const override {
                if (!supports(weight, scheme) ||
                    static_cast<int64_t>(values.size()) != Common::element_count(weight))
                    throw std::invalid_argument("Q4K: invalid Scheme, shape, or value count");
                Common::require_finite(values, storage_name());

                const std::size_t group_elements = static_cast<std::size_t>(GROUP_ELEMENTS);
                const std::size_t groups = values.size() / group_elements;
                std::vector<uint8_t> q((values.size() + 1) / 2, 0);
                std::vector<uint8_t> scales;
                std::vector<uint8_t> zeros;
                scales.reserve(groups * 2);
                zeros.reserve(groups * 2);

                for (std::size_t group = 0; group < groups; ++group) {
                    const float *source = values.data() + group * group_elements;
                    const auto [minimum, maximum] =
                        std::minmax_element(source, source + group_elements);
                    float scale = 0.0f;
                    float zero = 0.0f;
                    if (*maximum == *minimum) {
                        if (*minimum > 0.0f) {
                            scale = *minimum;
                            zero = 0.0f;
                        } else if (*minimum < 0.0f) {
                            scale = -*minimum;
                            zero = 1.0f;
                        } else {
                            scale = 1.0f;
                            zero = 0.0f;
                        }
                    } else {
                        const auto optimized = Search::grouped_affine_integer_zero(
                            std::span<const float>(source, group_elements),
                            static_cast<int>(MAXIMUM));
                        scale = optimized.scale;
                        zero = optimized.zero;
                    }
                    scale = Common::half_to_float(Common::float_to_half(scale));
                    zero = Common::half_to_float(Common::float_to_half(zero));
                    Common::append_half(scales, scale);
                    Common::append_half(zeros, zero);

                    for (std::size_t i = 0; i < group_elements; ++i) {
                        const long quantized =
                            std::clamp(static_cast<long>(std::nearbyint(source[i] / scale + zero)),
                                       MINIMUM, MAXIMUM);
                        const std::size_t index = group * group_elements + i;
                        if ((index & 1) == 0)
                            q[index / 2] = static_cast<uint8_t>(quantized);
                        else
                            q[index / 2] |= static_cast<uint8_t>(quantized << 4);
                    }
                }
                return {{".q", std::move(q)},
                        {".scale", std::move(scales)},
                        {".zero", std::move(zeros)}};
            }
    };
}
