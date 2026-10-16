#pragma once

#include "quant/common.hpp"

namespace Quantizations {
    class BF16 final : public Codec {
        public:
            BF16() { name = "BF16"; }

            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                Tensor raw = ops.insert<Functional::Const>(Weight{weight.name, weight.shape}, false,
                                                           DType::Int16);

                Tensor widened = ops.insert<Functional::Convert>(std::move(raw), DType::Int32);
                Tensor masked =
                    Common::mask(ops, std::move(widened), 0xFFFF, weight.shape, DType::Int32);
                Tensor shifted =
                    Common::shift_left(ops, std::move(masked), 16, weight.shape, DType::Int32);

                Tensor value = ops.insert<Functional::BitcastConvert>(std::move(shifted),
                                                                      DType::Float32, weight.shape);

                return Common::convert_if_needed(ops, std::move(value), DType::Float32, dtype);
            }

            std::vector<uint8_t> encode(std::span<const float> values) const override {
                Common::require_finite(values, name);
                std::vector<uint8_t> bytes(values.size() * 2);

                for (std::size_t i = 0; i < values.size(); ++i) {
                    const uint16_t half = Common::float_to_bfloat16(values[i]);
                    bytes[i * 2] = static_cast<uint8_t>(half);
                    bytes[i * 2 + 1] = static_cast<uint8_t>(half >> 8);
                }

                return bytes;
            }

            std::vector<float> decode(std::span<const uint8_t> bytes) const override {
                if (bytes.size() % 2 != 0) {
                    throw std::invalid_argument("BF16: size is not aligned to 2 bytes");
                }

                std::vector<float> values(bytes.size() / 2);

                for (std::size_t i = 0; i < values.size(); ++i) {
                    values[i] =
                        Common::bfloat16_to_float(static_cast<uint16_t>(bytes[i * 2]) |
                                                  (static_cast<uint16_t>(bytes[i * 2 + 1]) << 8));
                }

                return values;
            }
    };
}
