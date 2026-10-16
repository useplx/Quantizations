#pragma once

#include "quant/common.hpp"
#include "quant/search.hpp"

namespace Quantizations {
    class Q8 final : public Codec {
        public:
            Q8() { name = "Q8_GROUPED"; }

            bool supports(const Weight &weight) const override {
                return weight.shape.size() == 2 && weight.shape[1] > 0 &&
                       weight.shape[1] % GROUP == 0;
            }

            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                if (!supports(weight))
                    throw std::invalid_argument("Q8_GROUPED requer matriz com K divisivel por 32");
                const int64_t rows = weight.shape[0];
                const int64_t groups = weight.shape[1] / GROUP;
                const int64_t words = GROUP / CODES_PER_WORD;
                const std::vector<int64_t> packed_shape{rows, groups, words};
                const std::vector<int64_t> decoded_shape{rows, groups, words, CODES_PER_WORD};

                Tensor codes = ops.insert<Functional::Const>(
                    Weight{weight.name + ".q", packed_shape}, false, DType::Int32);
                Tensor shifts = ops.insert<Functional::Const>(
                    Weight{weight.name + ".shifts", {CODES_PER_WORD}}, false, DType::Int32);
                Tensor scale = ops.insert<Functional::Const>(
                    Weight{weight.name + ".scale", {rows, groups}}, false, DType::Float16);
                Tensor zero = ops.insert<Functional::Const>(
                    Weight{weight.name + ".zero", {rows, groups}}, false, DType::Float16);
                Tensor decoded = ops.insert<Functional::Map>(
                    std::vector<Functional::Map::Operand>{
                        {std::move(codes), DType::Int32, {0, 1, 2}},
                        {std::move(shifts), DType::Int32, {3}},
                        {std::move(scale), DType::Float16, {0, 1}},
                        {std::move(zero), DType::Float16, {0, 1}},
                    },
                    decoded_shape, dtype, [dtype](Operations &body, std::span<const Tensor> args) {
                        Tensor values =
                            body.insert<Functional::ShiftRightLogical>(args[0], args[1]);
                        values = body.insert<Functional::BitwiseAnd>(
                            std::move(values), Common::scalar(body, int64_t{0xff}, DType::Int32));
                        values = body.insert<Functional::Convert>(std::move(values), DType::UInt32);
                        values = body.insert<Functional::Convert>(std::move(values), dtype);
                        Tensor scale_value =
                            Common::convert_if_needed(body, args[2], DType::Float16, dtype);
                        Tensor zero_value =
                            Common::convert_if_needed(body, args[3], DType::Float16, dtype);
                        Tensor shifted = body.insert<Functional::Subtract>(std::move(values),
                                                                           std::move(zero_value));
                        return body.insert<Functional::Multiply>(std::move(shifted),
                                                                 std::move(scale_value));
                    });
                return ops.insert<Functional::Reshape>(std::move(decoded), weight.shape);
            }

            std::vector<uint8_t> encode(std::span<const float> values) const override {
                if (values.size() % static_cast<std::size_t>(GROUP) != 0)
                    throw std::invalid_argument("Q8_GROUPED: element count is not aligned to 32");
                Common::require_finite(values, name);

                std::vector<uint8_t> out(
                    values.size() / static_cast<std::size_t>(GROUP) * GROUP_BYTES, 0);
                for (std::size_t group = 0; group < values.size() / static_cast<std::size_t>(GROUP);
                     ++group) {
                    const float *source = values.data() + group * GROUP;
                    const Search::Affine fit =
                        Search::grouped_affine(std::span<const float>(source, GROUP), LEVELS);

                    const uint16_t scale_bits = Common::float_to_half(fit.scale);
                    const uint16_t zero_bits = Common::float_to_half(fit.zero);
                    const float scale = Common::half_to_float(scale_bits);
                    const float zero = Common::half_to_float(zero_bits);

                    uint8_t *target = out.data() + group * GROUP_BYTES;
                    for (std::size_t i = 0; i < GROUP; ++i) {
                        const long code = scale == 0.0f
                                              ? 0
                                              : std::clamp(std::lround(source[i] / scale + zero),
                                                           0L, static_cast<long>(LEVELS));
                        target[i] = static_cast<uint8_t>(code);
                    }
                    target[GROUP + 0] = static_cast<uint8_t>(scale_bits);
                    target[GROUP + 1] = static_cast<uint8_t>(scale_bits >> 8);
                    target[GROUP + 2] = static_cast<uint8_t>(zero_bits);
                    target[GROUP + 3] = static_cast<uint8_t>(zero_bits >> 8);
                }
                return out;
            }

            std::vector<Mln::Sdk::Codec::Parameter::Component>
            encode(std::span<const float> values, const Weight &weight,
                   const Mln::Sdk::Codec::Scheme &) const override {
                if (!supports(weight))
                    throw std::invalid_argument(
                        "Q8_GROUPED: shape does not support composite storage");
                const std::vector<uint8_t> packed = encode(values);
                const std::size_t groups = values.size() / static_cast<std::size_t>(GROUP);

                std::vector<uint8_t> codes(groups * GROUP);
                std::vector<uint8_t> scale(groups * 2);
                std::vector<uint8_t> zero(groups * 2);
                for (std::size_t group = 0; group < groups; ++group) {
                    const uint8_t *source = packed.data() + group * GROUP_BYTES;
                    std::memcpy(codes.data() + group * GROUP, source, GROUP);
                    std::memcpy(scale.data() + group * 2, source + GROUP, 2);
                    std::memcpy(zero.data() + group * 2, source + GROUP + 2, 2);
                }
                std::vector<uint8_t> shifts(CODES_PER_WORD * sizeof(int32_t));
                for (int32_t lane = 0; lane < CODES_PER_WORD; ++lane) {
                    const int32_t shift = lane * BITS;
                    std::memcpy(shifts.data() + static_cast<std::size_t>(lane) * sizeof(shift),
                                &shift, sizeof(shift));
                }
                return {{".q", std::move(codes)},
                        {".shifts", std::move(shifts)},
                        {".scale", std::move(scale)},
                        {".zero", std::move(zero)}};
            }

            std::vector<float> decode(std::span<const uint8_t> bytes) const override {
                if (bytes.size() % GROUP_BYTES != 0)
                    throw std::invalid_argument("Q8_GROUPED: size is not aligned to the group");
                const std::size_t groups = bytes.size() / GROUP_BYTES;
                std::vector<float> values(groups * GROUP);
                for (std::size_t group = 0; group < groups; ++group) {
                    const uint8_t *source = bytes.data() + group * GROUP_BYTES;
                    const float scale = Common::half_to_float(
                        static_cast<uint16_t>(source[GROUP] | (source[GROUP + 1] << 8)));
                    const float zero = Common::half_to_float(
                        static_cast<uint16_t>(source[GROUP + 2] | (source[GROUP + 3] << 8)));
                    for (std::size_t i = 0; i < GROUP; ++i)
                        values[group * GROUP + i] = (static_cast<float>(source[i]) - zero) * scale;
                }
                return values;
            }

        private:
            static constexpr int64_t GROUP = 32;
            static constexpr int64_t BITS = 8;
            static constexpr int64_t CODES_PER_WORD = 4;
            static constexpr std::size_t GROUP_BYTES = static_cast<std::size_t>(GROUP) + 4;
            static constexpr int LEVELS = 255;
    };
}
