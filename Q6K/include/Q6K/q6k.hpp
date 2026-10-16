#pragma once

#include "quant/common.hpp"

namespace Quantizations {
    class Q6K final : public Codec {
        public:
            Q6K() { name = "Q6_K"; }

            bool supports(const Weight &weight) const override {
                return weight.shape.size() >= 2 && Common::element_count(weight) % 256 == 0;
            }

            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                constexpr int64_t block_elements = 256;
                const int64_t count = Common::element_count(weight);

                if (count % block_elements != 0) {
                    throw std::invalid_argument("Q6_K: weight '" + weight.name +
                                                "' is not aligned to 256 elements");
                }

                const int64_t blocks = count / block_elements;
                const std::vector<int64_t> lanes{blocks, 2, 2, 2, 2, 16};

                Tensor low = ops.insert<Functional::Reshape>(
                    ops.insert<Functional::Const>(Weight{weight.name + ".ql", {blocks, 128}}, false,
                                                  DType::Int8),
                    std::vector<int64_t>{blocks, 2, 2, 2, 16});
                low = Common::broadcast(ops, std::move(low), lanes, {0, 1, 3, 4, 5});
                Tensor low_shift = Common::multiply_integer(
                    ops, ops.insert<Functional::Iota>(std::vector<int64_t>{2}, 0, DType::Int8), 4,
                    {2});
                low = Common::shift_right_by(
                    ops, std::move(low), Common::broadcast(ops, std::move(low_shift), lanes, {2}));
                low = Common::mask(ops, std::move(low), 15, lanes);

                Tensor high = ops.insert<Functional::Reshape>(
                    ops.insert<Functional::Const>(Weight{weight.name + ".qh", {blocks, 64}}, false,
                                                  DType::Int8),
                    std::vector<int64_t>{blocks, 2, 2, 16});
                high = Common::broadcast(ops, std::move(high), lanes, {0, 1, 4, 5});
                Tensor high_shift = ops.insert<Functional::Add>(
                    Common::broadcast(
                        ops,
                        Common::multiply_integer(
                            ops,
                            ops.insert<Functional::Iota>(std::vector<int64_t>{2}, 0, DType::Int8),
                            4, {2}),
                        lanes, {2}),
                    Common::broadcast(
                        ops,
                        Common::multiply_integer(
                            ops,
                            ops.insert<Functional::Iota>(std::vector<int64_t>{2}, 0, DType::Int8),
                            2, {2}),
                        lanes, {3}));
                high = Common::shift_right_by(ops, std::move(high), std::move(high_shift));
                high = Common::mask(ops, std::move(high), 3, lanes);
                high = Common::multiply_integer(ops, std::move(high), 16, lanes);

                Tensor q = ops.insert<Functional::Add>(std::move(low), std::move(high));
                q = ops.insert<Functional::Subtract>(
                    std::move(q), Common::broadcast(ops, Common::scalar(ops, 32), lanes, {}));
                q = ops.insert<Functional::Convert>(std::move(q), dtype);

                Tensor scales = ops.insert<Functional::Reshape>(
                    ops.insert<Functional::Const>(Weight{weight.name + ".scales", {blocks, 16}},
                                                  false, DType::Int8),
                    std::vector<int64_t>{blocks, 2, 2, 2, 2});
                scales = ops.insert<Functional::Convert>(std::move(scales), dtype);
                scales = Common::broadcast(ops, std::move(scales), lanes, {0, 1, 2, 3, 4});

                Tensor d = ops.insert<Functional::Const>(Weight{weight.name + ".d", {blocks}},
                                                         false, DType::Float16);
                d = Common::convert_if_needed(ops, std::move(d), DType::Float16, dtype);
                d = Common::broadcast(ops, std::move(d), lanes, {0});

                Tensor value = ops.insert<Functional::Multiply>(std::move(q), std::move(scales));
                value = ops.insert<Functional::Multiply>(std::move(value), std::move(d));
                value = ops.insert<Functional::Reshape>(
                    std::move(value),
                    std::vector<int64_t>{weight.shape[0], weight.shape[1] / 32, 32});
                return ops.insert<Functional::Reshape>(std::move(value), weight.shape);
            }

            std::vector<uint8_t> encode(std::span<const float> values) const override {
                constexpr std::size_t block_elements = 256;
                constexpr std::size_t block_bytes = 210;

                if (values.size() % block_elements != 0) {
                    throw std::invalid_argument("Q6_K: element count is not aligned to 256");
                }

                Common::require_finite(values, name);
                std::vector<uint8_t> out(values.size() / block_elements * block_bytes, 0);

                for (std::size_t block = 0; block < values.size() / block_elements; ++block) {
                    const float *source = values.data() + block * block_elements;
                    uint8_t *target = out.data() + block * block_bytes;
                    float group_scale[16]{};
                    float max_scale = 0.0f;

                    for (std::size_t group = 0; group < 16; ++group) {
                        float absolute_max = 0.0f;

                        for (std::size_t i = 0; i < 16; ++i) {
                            absolute_max = std::max(absolute_max, std::abs(source[group * 16 + i]));
                        }

                        group_scale[group] = absolute_max / 31.0f;
                        max_scale = std::max(max_scale, group_scale[group]);
                    }

                    float d = max_scale / 127.0f;
                    const uint16_t d_bits = Common::float_to_half(d);
                    d = Common::half_to_float(d_bits);
                    target[208] = static_cast<uint8_t>(d_bits);
                    target[209] = static_cast<uint8_t>(d_bits >> 8);

                    for (std::size_t group = 0; group < 16; ++group) {
                        const int scale = d == 0.0f

                                              ? 0

                                              : static_cast<int>(std::clamp(
                                                    std::lround(group_scale[group] / d), 1l, 127l));

                        target[192 + group] = static_cast<uint8_t>(static_cast<int8_t>(scale));
                        const std::size_t half = group / 8;
                        const std::size_t within = group % 8;
                        const std::size_t quarter = within / 2;
                        const std::size_t part = within % 2;

                        for (std::size_t i = 0; i < 16; ++i) {
                            const float divisor = d * scale;

                            const int q = divisor == 0.0f

                                              ? 0

                                              : static_cast<int>(std::clamp(
                                                    std::lround(source[group * 16 + i] / divisor),
                                                    -32l, 31l));

                            const uint8_t packed_q = static_cast<uint8_t>(q + 32);

                            const std::size_t ql_offset =
                                half * 64 + (quarter % 2) * 32 + part * 16 + i;

                            if (quarter < 2) {
                                target[ql_offset] = packed_q & 0x0f;
                            } else {
                                target[ql_offset] |= static_cast<uint8_t>((packed_q & 0x0f) << 4);
                            }

                            const std::size_t qh_offset = 128 + half * 32 + part * 16 + i;

                            target[qh_offset] |=
                                static_cast<uint8_t>(((packed_q >> 4) & 0x03) << (quarter * 2));
                        }
                    }
                }

                return out;
            }

            std::vector<Mln::Sdk::Codec::Parameter::Component>
            encode(std::span<const float> values, const Weight &weight,
                   const Mln::Sdk::Codec::Scheme &) const override {
                if (!supports(weight) ||
                    static_cast<int64_t>(values.size()) != Common::element_count(weight))
                    throw std::invalid_argument("Q6_K: invalid shape or value count");

                constexpr std::size_t block_bytes = 210;
                const std::vector<uint8_t> packed = encode(values);
                const std::size_t blocks = packed.size() / block_bytes;
                std::vector<uint8_t> ql(blocks * 128);
                std::vector<uint8_t> qh(blocks * 64);
                std::vector<uint8_t> scales(blocks * 16);
                std::vector<uint8_t> d(blocks * 2);
                for (std::size_t block = 0; block < blocks; ++block) {
                    const uint8_t *source = packed.data() + block * block_bytes;
                    std::copy_n(source, 128, ql.data() + block * 128);
                    std::copy_n(source + 128, 64, qh.data() + block * 64);
                    std::copy_n(source + 192, 16, scales.data() + block * 16);
                    std::copy_n(source + 208, 2, d.data() + block * 2);
                }
                return {{".ql", std::move(ql)},
                        {".qh", std::move(qh)},
                        {".scales", std::move(scales)},
                        {".d", std::move(d)}};
            }

            std::vector<float> decode(std::span<const uint8_t> bytes) const override {
                constexpr std::size_t block_bytes = 210;

                if (bytes.size() % block_bytes != 0) {
                    throw std::invalid_argument("Q6_K: size is not aligned to 210 bytes");
                }

                std::vector<float> values(bytes.size() / block_bytes * 256);

                for (std::size_t block = 0; block < bytes.size() / block_bytes; ++block) {
                    const uint8_t *source = bytes.data() + block * block_bytes;

                    const float d =
                        Common::half_to_float(static_cast<uint16_t>(source[208]) |
                                              (static_cast<uint16_t>(source[209]) << 8));

                    for (std::size_t group = 0; group < 16; ++group) {
                        const std::size_t half = group / 8;
                        const std::size_t within = group % 8;
                        const std::size_t quarter = within / 2;
                        const std::size_t part = within % 2;
                        const int scale = static_cast<int8_t>(source[192 + group]);

                        for (std::size_t i = 0; i < 16; ++i) {
                            const std::size_t ql_offset =
                                half * 64 + (quarter % 2) * 32 + part * 16 + i;

                            const uint8_t low = quarter < 2

                                                    ? source[ql_offset] & 0x0f

                                                    : source[ql_offset] >> 4;

                            const std::size_t qh_offset = 128 + half * 32 + part * 16 + i;

                            const uint8_t high =
                                static_cast<uint8_t>((source[qh_offset] >> (quarter * 2)) & 0x03);

                            const int q = static_cast<int>(low | (high << 4)) - 32;
                            values[block * 256 + group * 16 + i] = d * scale * q;
                        }
                    }
                }

                return values;
            }
    };
}
