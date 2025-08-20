#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace deepseek {

// ===== activation =====
inline float gelu_tanh(float x) {
  const float kAlpha = std::sqrt(2.0f / static_cast<float>(M_PI));
  return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + 0.044715f * x * x * x)));
}

inline float act_eval(int act_id, float x, const float* params, int nparam) {
  switch (act_id) {
    case 0: /*gelu*/
      return 0.5f * x * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)));
    case 1: /*gelu_new/tanh*/
      return gelu_tanh(x);
    case 2: /*gelu_fast*/
      return 0.5f * x *
          (1.0f + std::tanh(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
    case 3: /*quick_gelu*/
      return x / (1.0f + std::exp(-1.702f * x));
    case 4: /*silu/swish*/
      return x / (1.0f + std::exp(-x));
    case 5: /*relu*/
      return x > 0 ? x : 0;
    case 6: /*relu6*/ {
      float y = x > 0 ? x : 0;
      return y < 6 ? y : 6;
    }
    case 7: /*relu2*/ {
      float y = x > 0 ? x : 0;
      return y * y;
    }
    case 8: /*tanh*/
      return std::tanh(x);
    case 9: /*sigmoid*/
      return 1.0f / (1.0f + std::exp(-x));
    case 10: /*linear*/
      return x;
    case 11: /*mish*/ {
      float sp = std::log1p(std::exp(x));
      return x * std::tanh(sp);
    }
    case 12: /*leaky_relu*/ {
      float slope = (params && nparam >= 1) ? params[0] : 0.01f;
      return x >= 0 ? x : slope * x;
    }
    case 13: /*laplace*/ {
      float mu = (params && nparam >= 1) ? params[0] : 0.707107f;
      float sigma = (params && nparam >= 2) ? params[1] : 0.282095f;
      float z = (x - mu) / (sigma * std::sqrt(2.0f));
      return 0.5f * (1.0f + std::erf(z));
    }
    case 14: /*gelu_10*/ {
      float y = 0.5f * x * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)));
      if (y < -10.f)
        y = -10.f;
      if (y > 10.f)
        y = 10.f;
      return y;
    }
    default:
      return x;
  }
}

// ===== Dense GLU MLP =====
// X:[N,D], Wg/Wu:[I,D], Wd:[D,I], out:[N,D]
inline void mlp_glu_forward(
    const float* X,
    const float* Wg,
    const float* Bg,
    const float* Wu,
    const float* Bu,
    const float* Wd,
    const float* Bd,
    int64_t N,
    int64_t D,
    int64_t I,
    int act_id,
    const float* act_params,
    int nparam,
    float* Y) {
  std::vector<float> G(I), U(I);
  for (int64_t n = 0; n < N; ++n) {
    const float* x = X + n * D;
    // gate/up
    for (int64_t i = 0; i < I; ++i) {
      const float* wg = Wg + i * D;
      const float* wu = Wu + i * D;
      float sg = 0.f, su = 0.f;
      for (int64_t d = 0; d < D; ++d) {
        sg += wg[d] * x[d];
        su += wu[d] * x[d];
      }
      if (Bg)
        sg += Bg[i];
      if (Bu)
        su += Bu[i];
      G[i] = act_eval(act_id, sg, act_params, nparam);
      U[i] = su * G[i];
    }
    // down
    float* y = Y + n * D;
    for (int64_t d = 0; d < D; ++d) {
      const float* wd = Wd + d * I;
      float s = 0.f;
      for (int64_t i = 0; i < I; ++i)
        s += wd[i] * U[i];
      if (Bd)
        s += Bd[d];
      y[d] = s;
    }
  }
}

// ===== Gating (support greedy / group_limited_greedy / noaux_tc) =====
// hidden:[B,T,H], weight:[E,H] -> topk_idx:[N,K], topk_w:[N,K] (N=B*T)
inline void moe_gate_forward(
    const float* X,
    const float* W,
    const float* bias_opt,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t E,
    int64_t K,
    int scoring_id,
    int method_id,
    int64_t n_group,
    int64_t topk_group,
    bool norm_topk_prob,
    float routed_scale,
    /*out*/ int64_t* topk_idx,
    /*out*/ float* topk_w) {
  const int64_t N = B * T;
  std::vector<float> logits(E), scores(E);

  auto softmax = [&](std::vector<float>& v) {
    float m = v[0];
    for (int64_t e = 1; e < E; ++e)
      m = std::max(m, v[e]);
    float s = 0.f;
    for (int64_t e = 0; e < E; ++e) {
      v[e] = std::exp(v[e] - m);
      s += v[e];
    }
    for (int64_t e = 0; e < E; ++e)
      v[e] /= s;
  };

  for (int64_t n = 0; n < N; ++n) {
    const float* x = X + n * H;
    // logits = x @ W^T
    for (int64_t e = 0; e < E; ++e) {
      const float* w = W + e * H;
      float s = 0.f;
      for (int64_t h = 0; h < H; ++h)
        s += w[h] * x[h];
      logits[e] = s;
    }
    // scoring
    scores = logits;
    if (scoring_id == 0) { // softmax
      softmax(scores);
    } else { // sigmoid
      for (int64_t e = 0; e < E; ++e)
        scores[e] = 1.0f / (1.0f + std::exp(-scores[e]));
    }

    // "choice_scores" for selecting idx:
    // - greedy/group: directly use scores
    // - noaux_tc: scores_for_choice = scores + bias (only used for selecting
    // idx), weights still come from original scores
    const bool is_group =
        (method_id == 1 /*group_limited_greedy*/ ||
         method_id == 2 /*noaux_tc*/);
    const bool use_bias_for_choice = (method_id == 2 && bias_opt != nullptr);
    std::vector<float> choice = scores;
    if (use_bias_for_choice) {
      for (int64_t e = 0; e < E; ++e)
        choice[e] += bias_opt[e];
    }

    auto pick_topk_from = [&](const std::vector<int64_t>& cand) {
      std::vector<std::pair<float, int64_t>> v;
      v.reserve(cand.size());
      for (auto e : cand)
        v.emplace_back(choice[e], e);
      std::partial_sort(
          v.begin(), v.begin() + K, v.end(), [](auto& a, auto& b) {
            return a.first > b.first;
          });
      for (int64_t k = 0; k < K; ++k) {
        int64_t e = v[k].second;
        topk_idx[n * K + k] = e;
        topk_w[n * K + k] = scores[e]; // weights always come from original
                                       // scores (align with python)
      }
    };

    if (is_group) {
      const int64_t groups = std::max<int64_t>(1, n_group);
      const int64_t eg = E / groups;

      // select topk_group groups (use sum of top2 of "choice")
      std::vector<float> gscore(groups, 0.f);
      for (int64_t g = 0; g < groups; ++g) {
        float m1 = -1e30f, m2 = -1e30f;
        for (int64_t j = g * eg; j < (g + 1) * eg; ++j) {
          float v = choice[j];
          if (v > m1) {
            m2 = m1;
            m1 = v;
          } else if (v > m2) {
            m2 = v;
          }
        }
        gscore[g] = m1 + m2;
      }
      std::vector<int64_t> gidx(groups);
      for (int64_t g = 0; g < groups; ++g)
        gidx[g] = g;
      std::partial_sort(
          gidx.begin(),
          gidx.begin() + topk_group,
          gidx.end(),
          [&](int64_t a, int64_t b) { return gscore[a] > gscore[b]; });

      std::vector<int64_t> cand;
      cand.reserve(topk_group * eg);
      for (int64_t i = 0; i < topk_group; ++i) {
        int64_t g = gidx[i];
        for (int64_t j = g * eg; j < (g + 1) * eg; ++j)
          cand.push_back(j);
      }
      pick_topk_from(cand);
    } else {
      std::vector<int64_t> cand(E);
      for (int64_t e = 0; e < E; ++e)
        cand[e] = e;
      pick_topk_from(cand);
    }

    // 归一化 / scaling
    if (K > 1 && norm_topk_prob) {
      float s = 0.f;
      for (int64_t k = 0; k < K; ++k)
        s += topk_w[n * K + k];
      float scale = (s > 1e-20f) ? (routed_scale / s) : 0.f;
      for (int64_t k = 0; k < K; ++k)
        topk_w[n * K + k] *= scale;
    } else {
      for (int64_t k = 0; k < K; ++k)
        topk_w[n * K + k] *= routed_scale;
    }
  }
}

// ===== MoE forward (GLU experts aggregation) =====
// X:[B,T,D], topk_idx/topk_w:[N,K]; run K experts' GLU MLP for each token and
// sum with weights
inline void moe_forward_glu(
    const float* X,
    int64_t B,
    int64_t T,
    int64_t D,
    const int64_t* topk_idx,
    const float* topk_w,
    int64_t K,
    const float* const* Wg,
    const float* const* Bg,
    const float* const* Wu,
    const float* const* Bu,
    const float* const* Wd,
    const float* const* Bd,
    const int64_t* I_per_expert,
    int act_id,
    const float* act_params,
    int nparam,
    float* Y) {
  const int64_t N = B * T;
  for (int64_t n = 0; n < N; ++n) {
    const float* x = X + n * D;
    float* y = Y + n * D;
    // zero
    for (int64_t d = 0; d < D; ++d)
      y[d] = 0.f;

    for (int64_t kk = 0; kk < K; ++kk) {
      int64_t e = topk_idx[n * K + kk];
      float w = topk_w[n * K + kk];
      int64_t I = I_per_expert[e];

      // write single expert GLU MLP output to temporary, then weight to y
      std::vector<float> tmp(
          D); // minimal implementation; later can reuse buffer
      // directly reuse single sample logic of mlp_glu_forward (N=1)
      std::vector<float> out_one(D);
      mlp_glu_forward(
          /*X*/ x,
          /*Wg*/ Wg[e],
          /*Bg*/ Bg[e],
          /*Wu*/ Wu[e],
          /*Bu*/ Bu[e],
          /*Wd*/ Wd[e],
          /*Bd*/ Bd[e],
          /*N*/ 1,
          /*D*/ D,
          /*I*/ I,
          act_id,
          act_params,
          nparam,
          out_one.data());

      for (int64_t d = 0; d < D; ++d)
        y[d] += w * out_one[d];
    }
  }
}

} // namespace deepseek
