#pragma once

#include <mln/sdk/codec.hpp>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Quantizations {
    using Mln::DType;
    using Mln::Sdk::Codec;
    using Mln::Sdk::Operations;
    using Mln::Sdk::Tensor;
    using Mln::Sdk::Weight;
    namespace Functional = Mln::Sdk::Functional;

    namespace Common {
        static_assert(std::endian::native == std::endian::little, "os codecs de peso requerem "
                                                                  "uma plataforma little-endian");

        inline uint16_t float_to_half(float value) {
            const uint32_t bits = std::bit_cast<uint32_t>(value);
            const uint32_t sign = (bits >> 16) & 0x8000;
            const uint32_t exponent = (bits >> 23) & 0xff;
            const uint32_t mantissa = bits & 0x7fffff;

            if (exponent == 0xff) {
                return static_cast<uint16_t>(sign | 0x7c00 | (mantissa ? 0x0200 : 0));
            }

            int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;

            if (half_exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00);
            }

            if (half_exponent <= 0) {
                if (half_exponent < -10) {
                    return static_cast<uint16_t>(sign);
                }

                uint32_t rounded = (mantissa | 0x800000) >> (1 - half_exponent);
                rounded += 0x0fff + ((rounded >> 13) & 1);

                return static_cast<uint16_t>(sign | (rounded >> 13));
            }

            uint32_t rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);

            if (rounded & 0x800000) {
                rounded = 0;

                if (++half_exponent >= 31) {
                    return static_cast<uint16_t>(sign | 0x7c00);
                }
            }

            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) |
                                         (rounded >> 13));
        }

        inline float half_to_float(uint16_t value) {
            const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
            uint32_t exponent = (value >> 10) & 0x1f;
            uint32_t mantissa = value & 0x03ff;
            uint32_t bits = 0;

            if (exponent == 0) {
                if (mantissa == 0) {
                    bits = sign;
                } else {
                    int shift = 0;

                    while ((mantissa & 0x0400) == 0) {
                        mantissa <<= 1;
                        ++shift;
                    }

                    mantissa &= 0x03ff;

                    bits =
                        sign | (static_cast<uint32_t>(127 - 14 - shift) << 23) | (mantissa << 13);
                }
            } else if (exponent == 31) {
                bits = sign | 0x7f800000 | (mantissa << 13);
            } else {
                bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }

            return std::bit_cast<float>(bits);
        }

        inline uint16_t float_to_bfloat16(float value) {
            uint32_t bits = std::bit_cast<uint32_t>(value);

            if (std::isnan(value)) {
                return static_cast<uint16_t>(bits >> 16) | 0x0040;
            }

            const uint32_t rounding_bias = 0x7fff + ((bits >> 16) & 1);
            bits += rounding_bias;

            return static_cast<uint16_t>(bits >> 16);
        }

        inline float bfloat16_to_float(uint16_t value) {
            return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
        }

        inline void append_half(std::vector<uint8_t> &out, float value) {
            const uint16_t half = float_to_half(value);
            out.push_back(static_cast<uint8_t>(half));
            out.push_back(static_cast<uint8_t>(half >> 8));
        }

        inline float read_half(std::span<const uint8_t> bytes, std::size_t offset) {
            if (offset + 2 > bytes.size()) {
                throw std::invalid_argument("Quant: fp16 truncado");
            }

            return half_to_float(static_cast<uint16_t>(bytes[offset]) |
                                 (static_cast<uint16_t>(bytes[offset + 1]) << 8));
        }

        inline void require_finite(std::span<const float> values, std::string_view quant) {
            if (std::any_of(values.begin(), values.end(),
                            [](float value) { return !std::isfinite(value); })) {
                throw std::invalid_argument(std::string(quant) + ": pesos contem NaN "
                                                                 "ou infinito");
            }
        }

        inline int64_t element_count(const Weight &weight) {
            int64_t count = 1;

            for (int64_t dimension : weight.shape) {
                if (dimension <= 0 || count > INT64_MAX / dimension) {
                    throw std::invalid_argument("Quant: shape invalido "
                                                "em '" +
                                                weight.name + "'");
                }

                count *= dimension;
            }

            return count;
        }

        inline Tensor scalar(Operations &ops, int64_t value, DType dtype = DType::Int8) {
            return ops.insert<Functional::Const>(Weight{"", {}, static_cast<double>(value)}, false,
                                                 dtype);
        }

        inline Tensor scalar(Operations &ops, double value, DType dtype) {
            return ops.insert<Functional::Const>(Weight{"", {}, value}, false, dtype);
        }

        inline Tensor broadcast(Operations &ops, Tensor value, std::vector<int64_t> shape,
                                std::vector<int64_t> dimensions) {
            return ops.insert<Functional::Broadcast>(std::move(value), std::move(shape),
                                                     std::move(dimensions));
        }

        inline Tensor slice_columns(Operations &ops, Tensor value, int64_t rows, int64_t start,
                                    int64_t length) {
            return ops.insert<Functional::Slice>(std::move(value), std::vector<int64_t>{0, start},
                                                 std::vector<int64_t>{rows, start + length},
                                                 std::vector<int64_t>{1, 1});
        }

        inline Tensor column(Operations &ops, Tensor value, int64_t rows, int64_t index) {
            return ops.insert<Functional::Reshape>(
                slice_columns(ops, std::move(value), rows, index, 1), std::vector<int64_t>{rows});
        }

        inline Tensor mask(Operations &ops, Tensor value, int64_t bits,
                           const std::vector<int64_t> &shape, DType dtype = DType::Int8) {
            return ops.insert<Functional::BitwiseAnd>(
                std::move(value), broadcast(ops, scalar(ops, bits, dtype), shape, {}));
        }

        inline Tensor shift_right(Operations &ops, Tensor value, int64_t bits,
                                  const std::vector<int64_t> &shape, DType dtype = DType::Int8) {
            return ops.insert<Functional::ShiftRightLogical>(
                std::move(value), broadcast(ops, scalar(ops, bits, dtype), shape, {}));
        }

        inline Tensor shift_right_by(Operations &ops, Tensor value, Tensor amount) {
            return ops.insert<Functional::ShiftRightLogical>(std::move(value), std::move(amount));
        }

        inline Tensor shift_left(Operations &ops, Tensor value, int64_t bits,
                                 const std::vector<int64_t> &shape, DType dtype) {
            return ops.insert<Functional::ShiftLeftLogical>(
                std::move(value), broadcast(ops, scalar(ops, bits, dtype), shape, {}));
        }

        inline Tensor multiply_integer(Operations &ops, Tensor value, int64_t multiplier,
                                       const std::vector<int64_t> &shape) {
            return ops.insert<Functional::Multiply>(
                std::move(value), broadcast(ops, scalar(ops, multiplier), shape, {}));
        }

        inline Tensor convert_if_needed(Operations &ops, Tensor value, DType source, DType target) {
            return source == target

                       ? value

                       : ops.insert<Functional::Convert>(std::move(value), target);
        }

        inline Tensor decode_affine_grouped(Operations &ops, const Weight &weight,
                                            int64_t group_elements, int64_t bits, DType dtype) {
            const int64_t rows = weight.shape[0];
            const int64_t groups = weight.shape[1] / group_elements;
            const std::vector<int64_t> blocked{rows, groups, group_elements};

            Tensor codes = ops.insert<Functional::Const>(Weight{weight.name + ".q", blocked}, false,
                                                         bits == 4 ? DType::Int4 : DType::Int8);

            Tensor scale = ops.insert<Functional::Const>(
                Weight{weight.name + ".scale", {rows, groups}}, false, DType::Float16);
            Tensor zero = ops.insert<Functional::Const>(
                Weight{weight.name + ".zero", {rows, groups}}, false, DType::Float16);
            Tensor decoded = ops.insert<Functional::Map>(
                std::vector<Functional::Map::Operand>{
                    {std::move(codes), bits == 4 ? DType::Int4 : DType::Int8, {0, 1, 2}},
                    {std::move(scale), DType::Float16, {0, 1}},
                    {std::move(zero), DType::Float16, {0, 1}},
                },
                blocked, dtype, [dtype](Operations &body, std::span<const Tensor> args) {
                    Tensor values = body.insert<Functional::Convert>(args[0], DType::UInt32);
                    values = body.insert<Functional::Convert>(std::move(values), dtype);
                    Tensor scale_value = convert_if_needed(body, args[1], DType::Float16, dtype);
                    Tensor zero_value = convert_if_needed(body, args[2], DType::Float16, dtype);
                    Tensor shifted =
                        body.insert<Functional::Subtract>(std::move(values), std::move(zero_value));
                    return body.insert<Functional::Multiply>(std::move(shifted),
                                                             std::move(scale_value));
                });
            return ops.insert<Functional::Reshape>(std::move(decoded), weight.shape);
        }
    }
}
