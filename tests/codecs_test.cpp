#include "Q4K/q4k.hpp"
#include "Q5K/q5k.hpp"
#include "Q6K/q6k.hpp"
#include "Q8/q8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

namespace {
    using Bytes = std::vector<uint8_t>;

    std::vector<float> signal(std::size_t size) {
        std::vector<float> values(size);
        for (std::size_t i = 0; i < size; ++i) {
            const float x = static_cast<float>(i);
            values[i] = 2.7f * std::sin(x * 0.071f) + 0.8f * std::cos(x * 0.193f) +
                        static_cast<float>(static_cast<int>(i % 11) - 5) * 0.09f;
        }
        return values;
    }

    void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }

    void require_quality(std::string_view name, std::span<const float> source,
                         std::span<const float> decoded, double limit) {
        require(source.size() == decoded.size(), "decode changed the element count");
        double squared_error = 0.0;
        double squared_source = 0.0;
        for (std::size_t i = 0; i < source.size(); ++i) {
            const double error = static_cast<double>(decoded[i]) - source[i];
            squared_error += error * error;
            squared_source += static_cast<double>(source[i]) * source[i];
        }
        const double relative = squared_source == 0.0 ? std::sqrt(squared_error)
                                                      : std::sqrt(squared_error / squared_source);
        if (!std::isfinite(relative) || relative > limit)
            throw std::runtime_error(std::string(name) + ": relative RMS error " +
                                     std::to_string(relative));
        std::println("{} relative_rmse={}", name, relative);
    }

    uint16_t half_bits(float value) { return Quantizations::Common::float_to_half(value); }

    void put_half(Bytes &bytes, std::size_t offset, float value) {
        const uint16_t bits = half_bits(value);
        bytes[offset] = static_cast<uint8_t>(bits);
        bytes[offset + 1] = static_cast<uint8_t>(bits >> 8);
    }

    std::vector<float> reference_q5(std::span<const uint8_t> bytes) {
        require(bytes.size() % 176 == 0, "Q5 reference: invalid block");
        std::vector<float> output(bytes.size() / 176 * 256);
        for (std::size_t block = 0; block < bytes.size() / 176; ++block) {
            const uint8_t *raw = bytes.data() + block * 176;
            const float d =
                Quantizations::Common::half_to_float(uint16_t(raw[0]) | (uint16_t(raw[1]) << 8));
            const float dm =
                Quantizations::Common::half_to_float(uint16_t(raw[2]) | (uint16_t(raw[3]) << 8));
            for (std::size_t group = 0; group < 8; ++group) {
                const uint8_t scale =
                    group < 4 ? raw[4 + group] & 0x3f
                              : uint8_t((raw[8 + group] & 0x0f) | ((raw[group] >> 2) & 0x30));
                const uint8_t minimum =
                    group < 4 ? raw[8 + group] & 0x3f
                              : uint8_t((raw[8 + group] >> 4) | ((raw[4 + group] >> 2) & 0x30));
                for (std::size_t lane = 0; lane < 32; ++lane) {
                    const uint8_t packed = raw[48 + (group / 2) * 32 + lane];
                    const uint8_t low = group % 2 == 0 ? packed & 0x0f : packed >> 4;
                    const uint8_t high = (raw[16 + lane] >> group) & 1;
                    const float q = static_cast<float>(low | (high << 4));
                    output[block * 256 + group * 32 + lane] = d * scale * q - dm * minimum;
                }
            }
        }
        return output;
    }

    std::vector<float> reference_q6(std::span<const uint8_t> bytes) {
        require(bytes.size() % 210 == 0, "Q6 reference: invalid block");
        std::vector<float> output(bytes.size() / 210 * 256);
        for (std::size_t block = 0; block < bytes.size() / 210; ++block) {
            const uint8_t *raw = bytes.data() + block * 210;
            const float d = Quantizations::Common::half_to_float(uint16_t(raw[208]) |
                                                                 (uint16_t(raw[209]) << 8));
            for (std::size_t group = 0; group < 16; ++group) {
                const std::size_t half = group / 8;
                const std::size_t within = group % 8;
                const std::size_t quarter = within / 2;
                const std::size_t part = within % 2;
                const int scale = static_cast<int8_t>(raw[192 + group]);
                for (std::size_t lane = 0; lane < 16; ++lane) {
                    const std::size_t ql = half * 64 + (quarter % 2) * 32 + part * 16 + lane;
                    const uint8_t low = quarter < 2 ? raw[ql] & 0x0f : raw[ql] >> 4;
                    const std::size_t qh = 128 + half * 32 + part * 16 + lane;
                    const uint8_t high = (raw[qh] >> (quarter * 2)) & 3;
                    const int q = static_cast<int>(low | (high << 4)) - 32;
                    output[block * 256 + group * 16 + lane] = d * scale * q;
                }
            }
        }
        return output;
    }

    void require_equal(std::span<const float> lhs, std::span<const float> rhs,
                       std::string_view message) {
        require(lhs.size() == rhs.size(), message);
        for (std::size_t i = 0; i < lhs.size(); ++i)
            if (lhs[i] != rhs[i])
                throw std::runtime_error(std::string(message) + " at index " + std::to_string(i));
    }

    std::vector<float>
    decode_q4_components(const std::vector<Mln::Sdk::Codec::Parameter::Component> &components,
                         std::size_t count) {
        require(components.size() == 3, "Q4K did not produce three components");
        const Bytes *q = nullptr;
        const Bytes *scales = nullptr;
        const Bytes *zeros = nullptr;
        for (const auto &component : components) {
            if (component.suffix == ".q")
                q = &component.bytes;
            if (component.suffix == ".scale")
                scales = &component.bytes;
            if (component.suffix == ".zero")
                zeros = &component.bytes;
        }
        require(q && scales && zeros, "Q4K omitted a storage component");
        std::vector<float> output(count);
        for (std::size_t group = 0; group < count / 32; ++group) {
            const float scale = Quantizations::Common::half_to_float(
                uint16_t((*scales)[group * 2]) | (uint16_t((*scales)[group * 2 + 1]) << 8));
            const float zero = Quantizations::Common::half_to_float(
                uint16_t((*zeros)[group * 2]) | (uint16_t((*zeros)[group * 2 + 1]) << 8));
            for (std::size_t lane = 0; lane < 32; ++lane) {
                const uint8_t packed = (*q)[group * 16 + lane / 2];
                const uint8_t code = lane % 2 == 0 ? packed & 0x0f : packed >> 4;
                output[group * 32 + lane] = (static_cast<float>(code) - zero) * scale;
            }
        }
        return output;
    }

    void require_component_layout(
        std::span<const uint8_t> packed,
        const std::vector<Mln::Sdk::Codec::Parameter::Component> &components,
        std::size_t block_bytes,
        std::initializer_list<std::tuple<std::string_view, std::size_t, std::size_t>> fields) {
        require(packed.size() % block_bytes == 0, "composite storage: invalid block");
        require(components.size() == fields.size(), "composite storage: incorrect component count");
        const std::size_t blocks = packed.size() / block_bytes;
        for (const auto &[suffix, offset, width] : fields) {
            const auto found =
                std::find_if(components.begin(), components.end(),
                             [&](const auto &part) { return part.suffix == suffix; });
            require(found != components.end(), "composite storage: missing component");
            require(found->bytes.size() == blocks * width,
                    "composite storage: incorrect component size");
            for (std::size_t block = 0; block < blocks; ++block)
                require(std::equal(found->bytes.begin() + block * width,
                                   found->bytes.begin() + (block + 1) * width,
                                   packed.begin() + block * block_bytes + offset),
                        "storage composto diverge do encoding host");
        }
    }
}

int main() {
    using namespace Quantizations;

    const std::vector<float> values = signal(512);
    const Mln::Sdk::Weight matrix{"weight", {2, 256}};
    const Mln::Sdk::Codec::Scheme scheme{
        .scope = Mln::Sdk::Codec::Scope::Constant,
        .lifetime = Mln::Sdk::Codec::Lifetime::Immutable,
        .update = Mln::Sdk::Codec::Update::None,
    };

    Q4K q4;
    const auto q4_components = q4.encode(values, matrix, scheme);
    require(q4_components[0].bytes.size() == 256, "Q4K codes have an incorrect size");
    const auto &q4_zeros = q4_components[2].bytes;
    for (std::size_t offset = 0; offset < q4_zeros.size(); offset += 2) {
        const float zero = Quantizations::Common::half_to_float(
            uint16_t(q4_zeros[offset]) | (uint16_t(q4_zeros[offset + 1]) << 8));
        require(zero == std::round(zero), "Q4K zero point precisa ser inteiro");
    }
    require_quality("Q4K", values, decode_q4_components(q4_components, values.size()), 0.11);

    for (const float constant : {-0.375f, 0.0f, 0.625f}) {
        const std::vector<float> constant_values(512, constant);
        const auto components = q4.encode(constant_values, matrix, scheme);
        const auto decoded = decode_q4_components(components, constant_values.size());
        require_quality("Q4K constant", constant_values, decoded, 0.001);
        const auto &zeros = components[2].bytes;
        for (std::size_t offset = 0; offset < zeros.size(); offset += 2) {
            const float zero = Quantizations::Common::half_to_float(
                uint16_t(zeros[offset]) | (uint16_t(zeros[offset + 1]) << 8));
            require(zero == std::round(zero), "Q4K constant zero point precisa ser inteiro");
        }
    }

    Q5K q5;
    const Bytes q5_bytes = q5.encode(values);
    require(q5_bytes.size() == 352, "Q5K storage has an incorrect size");
    require_equal(q5.decode(q5_bytes), reference_q5(q5_bytes), "Q5K diverge do layout GGML");
    require_component_layout(
        q5_bytes, q5.encode(values, matrix, scheme), 176,
        {{".d", 0, 2}, {".dmin", 2, 2}, {".scales", 4, 12}, {".qh", 16, 32}, {".q", 48, 128}});
    require_quality("Q5K", values, q5.decode(q5_bytes), 0.07);

    Q6K q6;
    const Bytes q6_bytes = q6.encode(values);
    require(q6_bytes.size() == 420, "Q6K storage has an incorrect size");
    require_equal(q6.decode(q6_bytes), reference_q6(q6_bytes), "Q6K diverge do layout GGML");
    require_component_layout(
        q6_bytes, q6.encode(values, matrix, scheme), 210,
        {{".ql", 0, 128}, {".qh", 128, 64}, {".scales", 192, 16}, {".d", 208, 2}});
    require_quality("Q6K", values, q6.decode(q6_bytes), 0.04);

    Q8 q8;
    const Bytes q8_bytes = q8.encode(values);
    require(q8_bytes.size() == 576, "Q8 storage has an incorrect size");
    require_quality("Q8", values, q8.decode(q8_bytes), 0.012);

    const std::vector<float> zeros(512, 0.0f);
    require_quality("Q5K zero", zeros, q5.decode(q5.encode(zeros)), 0.0);
    require_quality("Q6K zero", zeros, q6.decode(q6.encode(zeros)), 0.0);
    require_quality("Q8 zero", zeros, q8.decode(q8.encode(zeros)), 0.0);

    Bytes q5_golden(176, 0);
    put_half(q5_golden, 0, 0.5f);
    put_half(q5_golden, 2, 0.25f);
    for (std::size_t i = 0; i < 12; ++i)
        q5_golden[4 + i] = uint8_t(i * 17 + 3);
    for (std::size_t i = 0; i < 32; ++i)
        q5_golden[16 + i] = uint8_t(i * 29 + 7);
    for (std::size_t i = 0; i < 128; ++i)
        q5_golden[48 + i] = uint8_t(i * 13 + 11);
    require_equal(q5.decode(q5_golden), reference_q5(q5_golden),
                  "Q5K failed the GGML golden vector");

    Bytes q6_golden(210, 0);
    for (std::size_t i = 0; i < 128; ++i)
        q6_golden[i] = uint8_t(i * 7 + 5);
    for (std::size_t i = 0; i < 64; ++i)
        q6_golden[128 + i] = uint8_t(i * 19 + 3);
    for (std::size_t i = 0; i < 16; ++i)
        q6_golden[192 + i] = static_cast<uint8_t>(static_cast<int8_t>(int(i) * 7 - 52));
    put_half(q6_golden, 208, 0.03125f);
    require_equal(q6.decode(q6_golden), reference_q6(q6_golden),
                  "Q6K failed the GGML golden vector");
}
