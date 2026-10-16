#pragma once

#include "quant/common.hpp"

namespace Quantizations {
    class F32 final : public Codec {
        public:
            F32() { name = "F32"; }

            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                Tensor value = ops.insert<Functional::Const>(weight, false, DType::Float32);

                return Common::convert_if_needed(ops, std::move(value), DType::Float32, dtype);
            }

            std::vector<uint8_t> encode(std::span<const float> values) const override {
                Common::require_finite(values, name);
                std::vector<uint8_t> bytes(values.size() * sizeof(float));

                if (!bytes.empty()) {
                    std::memcpy(bytes.data(), values.data(), bytes.size());
                }

                return bytes;
            }

            std::vector<float> decode(std::span<const uint8_t> bytes) const override {
                if (bytes.size() % sizeof(float) != 0) {
                    throw std::invalid_argument("F32: size is not aligned to 4 bytes");
                }

                std::vector<float> values(bytes.size() / sizeof(float));

                if (!bytes.empty()) {
                    std::memcpy(values.data(), bytes.data(), bytes.size());
                }

                return values;
            }
    };
}
