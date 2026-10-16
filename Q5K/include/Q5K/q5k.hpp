#pragma once

#include "quant/common.hpp"

namespace Quantizations {
    class Q5K final : public Codec {
        public:
            Q5K() { name = "Q5_K"; }

            bool supports(const Weight &weight) const override {
                return weight.shape.size() >= 2 && Common::element_count(weight) % 256 == 0;
            }

            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                constexpr int64_t block_elements = 256;
                const int64_t count = Common::element_count(weight);

                if (count % block_elements != 0) {
                    throw std::invalid_argument("Q5_K: weight '" + weight.name +
                                                "' is not aligned to 256 elements");
                }

                const int64_t blocks = count / block_elements;
                const std::vector<int64_t> index_shape{4, 2};
                const std::vector<int64_t> grouped{blocks, 4, 2};
                const std::vector<int64_t> lanes{blocks, 4, 2, 32};

                Tensor d = ops.insert<Functional::Const>(Weight{weight.name + ".d", {blocks}},
                                                         false, DType::Float16);
                Tensor dmin = ops.insert<Functional::Const>(Weight{weight.name + ".dmin", {blocks}},
                                                            false, DType::Float16);

                d = Common::convert_if_needed(ops, std::move(d), DType::Float16, dtype);
                dmin = Common::convert_if_needed(ops, std::move(dmin), DType::Float16, dtype);
                Tensor scales = ops.insert<Functional::Const>(
                    Weight{weight.name + ".scales", {blocks, 12}}, false, DType::Int8);
                Tensor qh = ops.insert<Functional::Const>(Weight{weight.name + ".qh", {blocks, 32}},
                                                          false, DType::Int8);
                Tensor quants = ops.insert<Functional::Const>(
                    Weight{weight.name + ".q", {blocks, 128}}, false, DType::Int8);

                const auto progression = [&](int64_t length, int64_t step, DType dtype_of) {
                    Tensor value =
                        ops.insert<Functional::Iota>(std::vector<int64_t>{length}, 0, dtype_of);
                    if (step == 1)
                        return value;
                    return ops.insert<Functional::Multiply>(
                        std::move(value),
                        Common::broadcast(ops, Common::scalar(ops, step, dtype_of),
                                          std::vector<int64_t>{length}, {}));
                };
                const auto constant = [&](int64_t value, DType dtype_of,
                                          const std::vector<int64_t> &shape) {
                    return Common::broadcast(ops, Common::scalar(ops, value, dtype_of), shape, {});
                };

                Tensor group = ops.insert<Functional::Add>(
                    Common::broadcast(ops, progression(4, 2, DType::Int32), index_shape, {0}),
                    Common::broadcast(ops, progression(2, 1, DType::Int32), index_shape, {1}));

                Tensor selector =
                    ops.insert<Functional::Divide>(group, constant(4, DType::Int32, index_shape));

                Tensor byte_at_group = ops.insert<Functional::Gather>(scales, group, 1);
                Tensor byte_above = ops.insert<Functional::Gather>(
                    scales,
                    ops.insert<Functional::Add>(group, constant(4, DType::Int32, index_shape)), 1);
                Tensor byte_below = ops.insert<Functional::Gather>(
                    scales,
                    ops.insert<Functional::Subtract>(
                        group, ops.insert<Functional::Multiply>(
                                   selector, constant(4, DType::Int32, index_shape))),
                    1);

                Tensor high_half = Common::broadcast(
                    ops, ops.insert<Functional::Convert>(std::move(selector), DType::Int8), grouped,
                    {1, 2});
                Tensor low_half =
                    ops.insert<Functional::Subtract>(constant(1, DType::Int8, grouped), high_half);

                Tensor scale = ops.insert<Functional::Add>(
                    ops.insert<Functional::Multiply>(low_half,
                                                     Common::mask(ops, byte_at_group, 63, grouped)),
                    ops.insert<Functional::Multiply>(
                        high_half, ops.insert<Functional::Add>(
                                       Common::mask(ops, byte_above, 15, grouped),
                                       Common::multiply_integer(
                                           ops, Common::shift_right(ops, byte_below, 6, grouped),
                                           16, grouped))));

                Tensor minimum = ops.insert<Functional::Add>(
                    ops.insert<Functional::Multiply>(low_half,
                                                     Common::mask(ops, byte_above, 63, grouped)),
                    ops.insert<Functional::Multiply>(
                        high_half, ops.insert<Functional::Add>(
                                       Common::shift_right(ops, byte_above, 4, grouped),
                                       Common::multiply_integer(
                                           ops, Common::shift_right(ops, byte_at_group, 6, grouped),
                                           16, grouped))));

                Tensor q =
                    Common::broadcast(ops,
                                      ops.insert<Functional::Reshape>(
                                          std::move(quants), std::vector<int64_t>{blocks, 4, 32}),
                                      lanes, {0, 1, 3});
                q = Common::mask(
                    ops,
                    Common::shift_right_by(
                        ops, std::move(q),
                        Common::broadcast(ops, progression(2, 4, DType::Int8), lanes, {2})),
                    15, lanes);

                Tensor high = Common::multiply_integer(
                    ops,
                    Common::mask(
                        ops,
                        Common::shift_right_by(
                            ops, Common::broadcast(ops, std::move(qh), lanes, {0, 3}),
                            ops.insert<Functional::Add>(
                                Common::broadcast(ops, progression(4, 2, DType::Int8), lanes, {1}),
                                Common::broadcast(ops, progression(2, 1, DType::Int8), lanes,
                                                  {2}))),
                        1, lanes),
                    16, lanes);

                q = ops.insert<Functional::Convert>(
                    ops.insert<Functional::Add>(std::move(q), std::move(high)), dtype);

                scale =
                    Common::broadcast(ops, ops.insert<Functional::Convert>(std::move(scale), dtype),
                                      lanes, {0, 1, 2});
                minimum = Common::broadcast(
                    ops, ops.insert<Functional::Convert>(std::move(minimum), dtype), lanes,
                    {0, 1, 2});

                Tensor value = ops.insert<Functional::Multiply>(
                    ops.insert<Functional::Multiply>(std::move(q), std::move(scale)),
                    Common::broadcast(ops, std::move(d), lanes, {0}));
                value = ops.insert<Functional::Subtract>(
                    std::move(value),
                    ops.insert<Functional::Multiply>(
                        std::move(minimum), Common::broadcast(ops, std::move(dmin), lanes, {0})));

                value = ops.insert<Functional::Reshape>(
                    std::move(value),
                    std::vector<int64_t>{weight.shape[0], weight.shape[1] / 32, 32});
                return ops.insert<Functional::Reshape>(std::move(value), weight.shape);
            }

            std::vector<uint8_t> encode(std::span<const float> values) const override {
                constexpr std::size_t block_elements = 256;
                constexpr std::size_t block_bytes = 176;

                if (values.size() % block_elements != 0) {
                    throw std::invalid_argument("Q5_K: element count is not aligned to 256");
                }

                Common::require_finite(values, name);
                std::vector<uint8_t> out(values.size() / block_elements * block_bytes, 0);

                for (std::size_t block = 0; block < values.size() / block_elements; ++block) {
                    const float *source = values.data() + block * block_elements;
                    uint8_t *target = out.data() + block * block_bytes;
                    float group_scale[8]{};
                    float group_minimum[8]{};
                    float max_scale = 0.0f;
                    float max_minimum = 0.0f;

                    for (std::size_t group = 0; group < 8; ++group) {
                        const float *begin = source + group * 32;
                        const auto [minimum, maximum] = std::minmax_element(begin, begin + 32);
                        group_minimum[group] = std::max(0.0f, -*minimum);
                        group_scale[group] = (*maximum + group_minimum[group]) / 31.0f;
                        max_scale = std::max(max_scale, group_scale[group]);
                        max_minimum = std::max(max_minimum, group_minimum[group]);
                    }

                    float d = max_scale / 63.0f;
                    float dmin = max_minimum / 63.0f;
                    const uint16_t d_bits = Common::float_to_half(d);
                    const uint16_t dmin_bits = Common::float_to_half(dmin);
                    d = Common::half_to_float(d_bits);
                    dmin = Common::half_to_float(dmin_bits);
                    target[0] = static_cast<uint8_t>(d_bits);
                    target[1] = static_cast<uint8_t>(d_bits >> 8);
                    target[2] = static_cast<uint8_t>(dmin_bits);
                    target[3] = static_cast<uint8_t>(dmin_bits >> 8);
                    uint8_t scales6[8]{};
                    uint8_t mins6[8]{};

                    for (std::size_t group = 0; group < 8; ++group) {
                        scales6[group] = d == 0.0f

                                             ? 0

                                             : static_cast<uint8_t>(std::clamp(
                                                   std::lround(group_scale[group] / d), 0l, 63l));

                        mins6[group] =
                            dmin == 0.0f

                                ? 0

                                : static_cast<uint8_t>(std::clamp(
                                      std::lround(group_minimum[group] / dmin), 0l, 63l));
                    }

                    for (std::size_t group = 0; group < 4; ++group) {
                        target[4 + group] = static_cast<uint8_t>((scales6[group] & 0x3f) |
                                                                 ((scales6[group + 4] >> 4) << 6));

                        target[8 + group] = static_cast<uint8_t>((mins6[group] & 0x3f) |
                                                                 ((mins6[group + 4] >> 4) << 6));

                        target[12 + group] = static_cast<uint8_t>((scales6[group + 4] & 0x0f) |
                                                                  ((mins6[group + 4] & 0x0f) << 4));
                    }

                    for (std::size_t group = 0; group < 8; ++group) {
                        const float scale = d * scales6[group];
                        const float minimum = dmin * mins6[group];

                        for (std::size_t i = 0; i < 32; ++i) {
                            const int q =
                                scale == 0.0f

                                    ? 0

                                    : static_cast<int>(std::clamp(
                                          std::lround((source[group * 32 + i] + minimum) / scale),
                                          0l, 31l));

                            const std::size_t q_offset = 48 + (group / 2) * 32 + i;

                            if ((group & 1) == 0) {
                                target[q_offset] = static_cast<uint8_t>(q & 0x0f);
                            } else {
                                target[q_offset] |= static_cast<uint8_t>((q & 0x0f) << 4);
                            }

                            if (q & 0x10) {
                                target[16 + i] |= static_cast<uint8_t>(1u << group);
                            }
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
                    throw std::invalid_argument("Q5_K: invalid shape or value count");

                constexpr std::size_t block_bytes = 176;
                const std::vector<uint8_t> packed = encode(values);
                const std::size_t blocks = packed.size() / block_bytes;
                std::vector<uint8_t> d(blocks * 2);
                std::vector<uint8_t> dmin(blocks * 2);
                std::vector<uint8_t> scales(blocks * 12);
                std::vector<uint8_t> qh(blocks * 32);
                std::vector<uint8_t> q(blocks * 128);
                for (std::size_t block = 0; block < blocks; ++block) {
                    const uint8_t *source = packed.data() + block * block_bytes;
                    std::copy_n(source, 2, d.data() + block * 2);
                    std::copy_n(source + 2, 2, dmin.data() + block * 2);
                    std::copy_n(source + 4, 12, scales.data() + block * 12);
                    std::copy_n(source + 16, 32, qh.data() + block * 32);
                    std::copy_n(source + 48, 128, q.data() + block * 128);
                }
                return {{".d", std::move(d)},
                        {".dmin", std::move(dmin)},
                        {".scales", std::move(scales)},
                        {".qh", std::move(qh)},
                        {".q", std::move(q)}};
            }

            std::vector<float> decode(std::span<const uint8_t> bytes) const override {
                constexpr std::size_t block_bytes = 176;

                if (bytes.size() % block_bytes != 0) {
                    throw std::invalid_argument("Q5_K: size is not aligned to 176 bytes");
                }

                std::vector<float> values(bytes.size() / block_bytes * 256);

                for (std::size_t block = 0; block < bytes.size() / block_bytes; ++block) {
                    const uint8_t *source = bytes.data() + block * block_bytes;

                    const float d = Common::half_to_float(static_cast<uint16_t>(source[0]) |
                                                          (static_cast<uint16_t>(source[1]) << 8));

                    const float dmin = Common::half_to_float(
                        static_cast<uint16_t>(source[2]) | (static_cast<uint16_t>(source[3]) << 8));

                    uint8_t scales6[8]{};
                    uint8_t mins6[8]{};

                    for (std::size_t group = 0; group < 4; ++group) {
                        scales6[group] = source[4 + group] & 0x3f;
                        mins6[group] = source[8 + group] & 0x3f;

                        scales6[group + 4] = static_cast<uint8_t>(((source[4 + group] >> 6) << 4) |
                                                                  (source[12 + group] & 0x0f));

                        mins6[group + 4] = static_cast<uint8_t>(((source[8 + group] >> 6) << 4) |
                                                                (source[12 + group] >> 4));
                    }

                    for (std::size_t group = 0; group < 8; ++group) {
                        const float scale = d * scales6[group];
                        const float minimum = dmin * mins6[group];

                        for (std::size_t i = 0; i < 32; ++i) {
                            const uint8_t packed = source[48 + (group / 2) * 32 + i];

                            const int low = (group & 1) == 0

                                                ? packed & 0x0f

                                                : packed >> 4;

                            const int high = ((source[16 + i] >> group) & 1) << 4;

                            values[block * 256 + group * 32 + i] =
                                scale * static_cast<float>(low | high) - minimum;
                        }
                    }
                }

                return values;
            }
    };
}
