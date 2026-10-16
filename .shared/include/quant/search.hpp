#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace Quantizations::Search {
    struct Affine {
            float scale = 0.0f;
            float zero = 0.0f;
    };

    inline Affine grouped_affine(std::span<const float> group, int levels) {
        const auto [low, high] = std::minmax_element(group.begin(), group.end());
        float minimum = *low;
        float maximum = *high;
        if (minimum > 0.0f)
            minimum = 0.0f;
        if (maximum == minimum)
            return {0.0f, 0.0f};

        float best_scale = (maximum - minimum) / static_cast<float>(levels);
        float best_zero = -minimum / best_scale;
        float best_error = -1.0f;

        constexpr int steps = 8;
        constexpr float SPAN = 0.12f;
        const float span = maximum - minimum;

        std::vector<int> codes(group.size());
        for (int step = 0; step <= steps; ++step) {
            const float factor =
                0.90f + (SPAN / static_cast<float>(steps)) * static_cast<float>(step);
            const float trial_scale = span * factor / static_cast<float>(levels);
            if (trial_scale <= 0.0f)
                continue;
            const float trial_zero = -minimum / trial_scale;

            double sum_q = 0.0, sum_qq = 0.0, sum_v = 0.0, sum_qv = 0.0;
            for (std::size_t i = 0; i < group.size(); ++i) {
                const int code =
                    static_cast<int>(std::clamp(std::lround(group[i] / trial_scale + trial_zero),
                                                0L, static_cast<long>(levels)));
                codes[i] = code;
                const double q = code;
                sum_q += q;
                sum_qq += q * q;
                sum_v += group[i];
                sum_qv += q * group[i];
            }

            const double count = static_cast<double>(group.size());
            const double determinant = count * sum_qq - sum_q * sum_q;
            if (determinant == 0.0)
                continue;
            const double scale = (count * sum_qv - sum_q * sum_v) / determinant;
            const double offset = (sum_qq * sum_v - sum_q * sum_qv) / determinant;
            if (scale <= 0.0)
                continue;

            double error = 0.0;
            for (std::size_t i = 0; i < group.size(); ++i) {
                const double difference = scale * codes[i] + offset - group[i];
                error += difference * difference;
            }
            if (best_error < 0.0 || error < best_error) {
                best_error = error;
                best_scale = static_cast<float>(scale);
                best_zero = static_cast<float>(-offset / scale);
            }
        }
        return {best_scale, best_zero};
    }

    inline Affine grouped_affine_integer_zero(std::span<const float> group, int levels) {
        const auto [low, high] = std::minmax_element(group.begin(), group.end());
        float minimum = std::min(*low, 0.0f);
        const float maximum = *high;
        if (maximum == minimum)
            return {1.0f, static_cast<float>(
                              std::clamp(std::lround(-minimum), 0L, static_cast<long>(levels)))};

        Affine best{};
        double best_error = -1.0;
        constexpr int steps = 8;
        constexpr float span = 0.12f;
        std::vector<int> codes(group.size());
        for (int step = 0; step <= steps; ++step) {
            const float factor = 0.90f + (span / static_cast<float>(steps)) * step;
            const float trial_scale = (maximum - minimum) * factor / levels;
            if (trial_scale <= 0.0f)
                continue;
            const int zero = static_cast<int>(
                std::clamp(std::lround(-minimum / trial_scale), 0L, static_cast<long>(levels)));
            for (std::size_t index = 0; index < group.size(); ++index)
                codes[index] = static_cast<int>(std::clamp(
                    std::lround(group[index] / trial_scale + zero), 0L, static_cast<long>(levels)));

            double numerator = 0.0;
            double denominator = 0.0;
            for (std::size_t index = 0; index < group.size(); ++index) {
                const double centered = codes[index] - zero;
                numerator += centered * group[index];
                denominator += centered * centered;
            }
            if (denominator == 0.0)
                continue;
            const double scale = numerator / denominator;
            if (scale <= 0.0)
                continue;
            double error = 0.0;
            for (std::size_t index = 0; index < group.size(); ++index) {
                const double difference = scale * (codes[index] - zero) - group[index];
                error += difference * difference;
            }
            if (best_error < 0.0 || error < best_error) {
                best_error = error;
                best = {static_cast<float>(scale), static_cast<float>(zero)};
            }
        }
        return best;
    }
}
