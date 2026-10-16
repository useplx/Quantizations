#pragma once

#include "Q5K/q5k.hpp"

namespace Quantizations {
    class KBlock final : public Codec {
        private:
            using MQ = Mln::Sdk::Codec;
            Q5K q5_;

        public:
            KBlock() { name = "K_BLOCK"; }

            std::string storage_name() const override { return "Q5_K"; }
            bool supports(const Weight &weight) const override { return q5_.supports(weight); }
            bool supports(const Weight &weight, const MQ::Scheme &) const override {
                return q5_.supports(weight);
            }
            Tensor decode(Operations &ops, const Weight &weight, DType dtype) override {
                return q5_.decode(ops, weight, dtype);
            }
            Tensor decode(Operations &ops, const Weight &weight, const MQ::Scheme &,
                          DType dtype) override {
                return q5_.decode(ops, weight, dtype);
            }
            std::vector<uint8_t> encode(std::span<const float> values) const override {
                return q5_.encode(values);
            }
            std::vector<uint8_t> encode(std::span<const float> values,
                                        const MQ::Scheme &) const override {
                return q5_.encode(values);
            }
            std::vector<float> decode(std::span<const uint8_t> bytes) const override {
                return q5_.decode(bytes);
            }
            std::vector<float> decode(std::span<const uint8_t> bytes,
                                      const MQ::Scheme &) const override {
                return q5_.decode(bytes);
            }
    };
}
