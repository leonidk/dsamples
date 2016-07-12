#pragma once

#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <cstring>
#include <random>
#include <array>

namespace ferns {

    static float clamp_f(float min, float max, float x)
    {
        return std::max(min, std::min(max, x));
    }


    template <typename T, int C>
    struct Image {
        std::shared_ptr<T> data;
        int width, height;
        T * ptr;
        Image()
            : data(nullptr)
            , width(0)
            , height(0)
            , ptr(nullptr)
        {
        }
        Image(int width, int height)
            : data(new T[width * height * C], arr_d())
            , width(width)
            , height(height)
        {
            ptr = data.get();
        }
        Image(int width, int height, T* d)
            : data(d, null_d())
            , width(width)
            , height(height)
            , ptr(d)
        {
        }

        struct null_d {
            void operator()(T const* p) {}
        };
        struct arr_d {
            void operator()(T const* p) { delete[] p; }
        };
        inline T sample(const float x, const float y, const int chan)
        {
            auto pixX = [this](float x) { return (int)clamp_f(0.0f, (float)(width - 1), std::round(x)); };
            auto pixY = [this](float y) { return (int)clamp_f(0.0f, (float)(height - 1), std::round(y)); };

            auto xm = pixX(x - 0.5f);
            auto xp = pixX(x + 0.5f);
            auto ym = pixY(y - 0.5f);
            auto yp = pixY(y + 0.5f);
            auto ptr = data.get();

            auto tl = ptr[C * (ym * width + xm) + chan];
            auto tr = ptr[C * (ym * width + xp) + chan];
            auto bl = ptr[C * (yp * width + xm) + chan];
            auto br = ptr[C * (yp * width + xp) + chan];

            float dx = x - xm;
            float dy = y - ym;

            auto sample = tl * (1.f - dx) * (1.f - dy) + tr * dx * (1.f - dy) + bl * (1.f - dx) * dy + br * dx * dy;
            return (T)sample;
        }
        Image<T, C> copy()
        {
            img::Image<T, C> res(width, height);
            memcpy(res.data.get(), this->data.get(), width*height*sizeof(T)*C);
            res.ptr = res.data.get();
            return res;
        }
    };

    template <typename T>
    using Img = Image<T, 1>;

    const int C = 2;
    class FernClassifier {
    public:
        FernClassifier(int fernSize, int numFerns, int numClasses, int w, int h)
            : fernSize(fernSize), numFerns(numFerns), numClasses(numClasses), twoFernSize(1 << (fernSize)),
            probs((1 << (fernSize))*(numClasses)*(numFerns), 1),
            counts((numClasses), (float)(1 << (fernSize - 1))),
            features(fernSize*numFerns), gen(std::random_device()()), w(w), h(h)
        {
        }
        void sampleFeatureFerns()
        {
            std::uniform_int_distribution<int> wDist(0, w - 1);
            std::uniform_int_distribution<int> hDist(0, h - 1);
            std::uniform_int_distribution<int> cDist(0, C - 1);

            for (int f = 0; f < numFerns; f++) {
                for (int d = 0; d < fernSize; d++) {
                    features[f*fernSize + d] = std::array<int32_t, 5>{{cDist(gen), wDist(gen), hDist(gen), wDist(gen), hDist(gen)}};
                }
            }
            probs = std::vector<float>(twoFernSize*(numClasses)*(numFerns), 1);
            counts = std::vector<float>((numClasses), (float)(1 << (fernSize - 1)));
        }
        void sampleOneFern()
        {
            std::uniform_int_distribution<int> wDist(0, w - 1);
            std::uniform_int_distribution<int> hDist(0, h - 1);
            std::uniform_int_distribution<int> cDist(0, C - 1);
            std::uniform_int_distribution<int> fDist(0, numFerns - 1);
            auto f = fDist(gen);
            for (int d = 0; d < fernSize; d++) {
                features[f*fernSize + d] = std::array<int32_t, 5>{{cDist(gen), wDist(gen), hDist(gen), wDist(gen), hDist(gen)}};
            }

            probs = std::vector<float>(twoFernSize*(numClasses)*(numFerns), 1);
            counts = std::vector<float>((numClasses), (float)(1 << (fernSize - 1)));
        }
        void sampleOneFeature()
        {
            std::uniform_int_distribution<int> wDist(0, w - 1);
            std::uniform_int_distribution<int> hDist(0, h - 1);
            std::uniform_int_distribution<int> cDist(0, C - 1);
            std::uniform_int_distribution<int> fDist(0, numFerns - 1);
            std::uniform_int_distribution<int> ftDist(0, fernSize - 1);
            std::bernoulli_distribution ft(0.5);

            auto f = fDist(gen);
            auto d = ftDist(gen);
            auto which = ft(gen);
            auto curr = features[f*fernSize + d];
            features[f*fernSize + d] = which ?
                std::array<int32_t, 5>{{cDist(gen), curr[1], curr[2], wDist(gen), hDist(gen)}} :
                std::array<int32_t, 5>{{cDist(gen), wDist(gen), hDist(gen), curr[3], curr[4]}};

            probs = std::vector<float>(twoFernSize*(numClasses)*(numFerns), 1);
            counts = std::vector<float>((numClasses), (float)(1 << (fernSize - 1)));
        }
        void sampleBadFeatures()
        {
            std::uniform_int_distribution<int> wDist(0, w - 1);
            std::uniform_int_distribution<int> hDist(0, h - 1);
            std::uniform_int_distribution<int> cDist(0, C - 1);
            std::bernoulli_distribution ft(0.5);

            std::vector<float> dotProducts(numFerns*fernSize*numClasses, 0.f);
            std::vector<float> dotProductsSum(numFerns*fernSize, 0.f);
            static float maxSum = 0;
            for (int f = 0; f < numFerns; f++) {
                for (int b = 0; b < fernSize; b++) {
                    for (int c = 0; c < numClasses; c++) {
                        auto bitMask = (1 << b);
                        for (int n = 0; n < twoFernSize; n++) {
                            if (n & bitMask) {
                                dotProducts[f*fernSize*numClasses + b*numClasses + c]
                                    += probs[f*(numClasses*twoFernSize) + c*twoFernSize + n];
                            }
                        }
                    }
                    for (int c = 0; c < numClasses; c++) {
                        dotProducts[f*fernSize*numClasses + b*numClasses + c] /= (counts[c] * numClasses);
                    }
                }
            }

            for (int f = 0; f < numFerns; f++) {
                for (int b = 0; b < fernSize; b++) {
                    // add up all differences between classes for a given feature

                    for (int r = 0; r < numClasses; r++) { //ref classes
                        auto ref = dotProducts[f*fernSize*numClasses + b*numClasses + r];
                        for (int t = r + 1; t < numClasses; t++) {
                            auto target = dotProducts[f*fernSize*numClasses + b*numClasses + t];
                            dotProductsSum[f*fernSize + b] += target*ref;
                        }
                    }
                }
            }

            float maxVal = *std::min_element(dotProductsSum.begin(), dotProductsSum.end());
            for (int f = 0; f < numFerns; f++) {
                for (int b = 0; b < fernSize; b++) {
                    auto v = dotProductsSum[f*fernSize + b];
                    if (v <= maxVal) {
                        auto which = ft(gen);
                        auto curr = features[f*fernSize + b];
                        features[f*fernSize + b] = which ?
                            std::array<int32_t, 5>{{cDist(gen), curr[1], curr[2], wDist(gen), hDist(gen)}} :
                            std::array<int32_t, 5>{{cDist(gen), wDist(gen), hDist(gen), curr[3], curr[4]}};
                        auto newft = features[f*fernSize + b];
                        features[f*fernSize + b] = newft;
                    }
                }
            }
            probs = std::vector<float>(twoFernSize*(numClasses)*(numFerns), 1);
            counts = std::vector<float>((numClasses), (float)(1 << (fernSize - 1)));
        }
        template<typename T>
        int getHash(const ferns::Image<T,C> & img, const int fern) {
            int hash = 0;
            for (int l = 0; l < fernSize; l++) {
                auto feature = features[fern*fernSize + l];
                auto c = feature[0];
                auto p1x = feature[1];
                auto p1y = feature[2];
                auto p2x = feature[3];
                auto p2y = feature[4];
                int bit = (img.ptr[c*(img.width*img.height) + (img.width*p1y + p1x)] > img.ptr[c*(img.width*img.height)+(img.width*p2y + p2x)]) ? 1 : 0;
                hash |= (bit << l);
            }
            return hash;
        }
        template<typename T>
        void train(const ferns::Image<T, C> & img, const uint8_t label) {
            for (int f = 0; f < numFerns; f++) {
                auto hash = getHash(img, f);
                probs[f*(numClasses*twoFernSize) + label*twoFernSize + hash]++;
            }
            counts[label]++;

        }
        template<typename T>
        std::vector<float> predict(const ferns::Image<T, C> & img) {
            std::vector<float> fernClassProbs(numFerns*numClasses, 0); //tem
            std::vector<float> fernSumProbs(numFerns, 0); //nrmzs

            std::vector<float> classProbs(numClasses, 0);

            for (int f = 0; f < numFerns; f++) {
                auto hash = getHash(img, f);
                for (int c = 0; c < numClasses; c++) {
                    auto probF_C = static_cast<float>(probs[f*(numClasses*twoFernSize) + c*twoFernSize + hash]) / static_cast<float>(counts[c]);
                    fernClassProbs[f*numClasses + c] = probF_C;
                    fernSumProbs[f] += probF_C;
                }
            }
            for (int f = 0; f < numFerns; f++) {
                for (int c = 0; c < numClasses; c++) {
                    fernClassProbs[f*numClasses + c] /= fernSumProbs[f];
                }
            }
            for (int c = 0; c < numClasses; c++) {
                auto prob = 0.0f;
                for (int f = 0; f < numFerns; f++) {
                    prob += logf(fernClassProbs[f*numClasses + c]);
                }
                classProbs[c] = prob;
            }
            auto max_value = *std::max_element(classProbs.begin(), classProbs.end());
            float sum = 0;
            for (auto & v : classProbs)
                sum += expf(v - max_value);
            for (auto & v : classProbs)
                v = expf(v - max_value) / sum;
            return classProbs;
        }

    private:
        std::vector<float> probs; // 2^fernSize x numClasses x numFerns
        std::vector<float> counts; // numClasses
        std::vector<std::array<int32_t,5>> features; // fernSize x numFerns. i(0,1) > i(2,3)
        std::mt19937 gen;
        int fernSize, numFerns, numClasses, twoFernSize;
        int w, h;
    };
}