#pragma once

#include "quant/common.hpp"

namespace Quantizations {
    class Int8 final : public Codec {
        private:
            using MQ = Mln::Sdk::Codec;

        public:
            Int8() { name = "INT8"; }

            bool supports(const Weight &weight) const override {
                return !weight.shape.empty() && weight.shape.front() > 0 &&
                       Common::element_count(weight) % weight.shape.front() == 0;
            }

            bool supports(const Weight &weight, const MQ::Scheme &) const override {
                return supports(weight);
            }

            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                return decode(ops, weight, MQ::Scheme{}, dtype);
            }

            Tensor decode(Operations &ops, const Weight &weight, const MQ::Scheme &,
                          DType dtype) override {
                if (!supports(weight))
                    throw std::invalid_argument("INT8: unsupported shape");
                const int64_t rows = weight.shape.front();
                Tensor values = ops.insert<Functional::Const>(
                    Weight{weight.name + ".q", weight.shape}, false, DType::Int8);
                values = Common::convert_if_needed(ops, std::move(values), DType::Int8, dtype);
                Tensor scale = ops.insert<Functional::Const>(Weight{weight.name + ".scale", {rows}},
                                                             false, DType::Float32);
                scale = Common::convert_if_needed(ops, std::move(scale), DType::Float32, dtype);
                scale = Common::broadcast(ops, std::move(scale), weight.shape, {0});
                return ops.insert<Functional::Multiply>(std::move(values), std::move(scale));
            }

            std::vector<Mln::Sdk::Codec::Parameter::Component>
            encode(std::span<const float> values, const Weight &weight,
                   const MQ::Scheme &) const override {
                if (!supports(weight) ||
                    static_cast<int64_t>(values.size()) != Common::element_count(weight))
                    throw std::invalid_argument("INT8: invalid shape or value count");
                Common::require_finite(values, name);
                const std::size_t rows = static_cast<std::size_t>(weight.shape.front());
                const std::size_t row_size = values.size() / rows;
                std::vector<uint8_t> q(values.size());
                std::vector<uint8_t> scales(rows * sizeof(float));
                for (std::size_t row = 0; row < rows; ++row) {
                    const float *begin = values.data() + row * row_size;
                    float max_abs = 0.0f;
                    for (std::size_t i = 0; i < row_size; ++i)
                        max_abs = std::max(max_abs, std::abs(begin[i]));
                    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
                    std::memcpy(scales.data() + row * sizeof(float), &scale, sizeof(float));
                    for (std::size_t i = 0; i < row_size; ++i) {
                        const long rounded = static_cast<long>(std::nearbyint(begin[i] / scale));
                        const auto quantized =
                            static_cast<int8_t>(std::clamp(rounded, -127L, 127L));
                        q[row * row_size + i] = std::bit_cast<uint8_t>(quantized);
                    }
                }
                return {{".q", std::move(q)}, {".scale", std::move(scales)}};
            }
    };
}
