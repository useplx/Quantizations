#pragma once

#include "quant/common.hpp"

namespace Quantizations {
    class F16 final : public Codec {
        public:
            F16() { name = "F16"; }

            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                Tensor value = ops.insert<Functional::Const>(weight, false, DType::Float16);

                return Common::convert_if_needed(ops, std::move(value), DType::Float16, dtype);
            }

            std::vector<uint8_t> encode(std::span<const float> values) const override {
                Common::require_finite(values, name);
                std::vector<uint8_t> bytes;
                bytes.reserve(values.size() * 2);

                for (float value : values) {
                    Common::append_half(bytes, value);
                }

                return bytes;
            }

            std::vector<float> decode(std::span<const uint8_t> bytes) const override {
                if (bytes.size() % 2 != 0) {
                    throw std::invalid_argument("F16: size is not aligned to 2 bytes");
                }

                std::vector<float> values(bytes.size() / 2);

                for (std::size_t i = 0; i < values.size(); ++i) {
                    values[i] = Common::read_half(bytes, i * 2);
                }

                return values;
            }
    };
}
