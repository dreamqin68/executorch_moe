#ifndef EXECUTORCH_ENABLE_LOGGING
#define EXECUTORCH_ENABLE_LOGGING 0
#endif

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/operator_registry.h>

#include "../common/moe_core.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <utility> // for std::move
#include <vector>

using executorch::aten::Tensor;
using executorch::aten::TensorList;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;

// Dtype guard helper
static inline bool is_f32(const Tensor& t) {
  using executorch::aten::ScalarType;
  return t.scalar_type() == ScalarType::Float;
}

// ---- convenient data_ptr wrapper ----
static inline const float* cf32(const Tensor& t) {
  return t.const_data_ptr<float>();
}
static inline float* f32(Tensor& t) {
  return t.mutable_data_ptr<float>();
}
static inline const int64_t* ci64(const Tensor& t) {
  return t.const_data_ptr<int64_t>();
}
static inline int64_t* i64(Tensor& t) {
  return t.mutable_data_ptr<int64_t>();
}

//=========================
// 1) Dense GLU MLP (out)
//=========================
Tensor& mlp_glu_out(
    KernelRuntimeContext& ctx,
    const Tensor& x, // [N, Din]
    const Tensor& Wg, // [I, Din]
    const Tensor& Bg, // [I] or empty
    const Tensor& Wu, // [I, Din]
    const Tensor& Bu, // [I] or empty
    const Tensor& Wd, // [Din, I]
    const Tensor& Bd, // [Din] or empty
    int64_t act_id, // activation ID (align with moe_core.h)
    const Tensor& act_params, // allow empty, custom activation parameters
    Tensor& out) { // [N, Din]

  fprintf(stderr, "[moe] ENTER %s\n", __func__);
  fflush(stderr);

  // Dtype guard check
  if (!is_f32(x) || !is_f32(Wg) || !is_f32(Wu) || !is_f32(Wd)) {
    fprintf(
        stderr,
        "[moe] %s expects FP32. x=%d Wg=%d Wu=%d Wd=%d\n",
        __func__,
        (int)x.scalar_type(),
        (int)Wg.scalar_type(),
        (int)Wu.scalar_type(),
        (int)Wd.scalar_type());
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  // light parameter check
  if (x.dim() != 2 || Wg.dim() != 2 || Wu.dim() != 2 || Wd.dim() != 2) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  if (Wg.size(1) != x.size(1) || Wu.size(1) != x.size(1) ||
      Wd.size(1) != Wg.size(0)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  if (out.dim() != 2 || out.size(0) != x.size(0) || out.size(1) != x.size(1)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  const float* ap = act_params.numel() ? cf32(act_params) : nullptr;
  const int nparam =
      act_params.numel() ? static_cast<int>(act_params.numel()) : 0;

  deepseek::mlp_glu_forward(
      /*x*/ cf32(x),
      /*Wg*/ cf32(Wg),
      /*Bg*/ (Bg.numel() ? cf32(Bg) : nullptr),
      /*Wu*/ cf32(Wu),
      /*Bu*/ (Bu.numel() ? cf32(Bu) : nullptr),
      /*Wd*/ cf32(Wd),
      /*Bd*/ (Bd.numel() ? cf32(Bd) : nullptr),
      /*N*/ x.size(0),
      /*Din*/ x.size(1),
      /*I*/ Wg.size(0),
      /*act_id*/ static_cast<int>(act_id),
      /*act_p*/ ap,
      /*np*/ nparam,
      /*out*/ f32(out));

  fprintf(stderr, "[moe] EXIT %s ok\n", __func__);
  fflush(stderr);

  return out;
}

//======================================
// 2) Gate selection (two out Tensor: topk_idx, topk_w)
//======================================
void moe_gate_forward_out(
    KernelRuntimeContext& ctx,
    const Tensor& hidden, // [B, T, H]
    const Tensor& weight, // [E, H]
    int64_t top_k,
    double routed_scaling_factor, //  double receive, then cast to float
    bool norm_topk_prob,
    int64_t scoring_id,
    int64_t method_id,
    int64_t n_group,
    int64_t topk_group,
    bool /*training*/,
    double alpha, // double receive, then cast to float
    bool /*seq_aux*/,
    const Tensor& e_score_correction_bias, // empty tensor means None
    Tensor& topk_idx_out, // expect int64
    Tensor& topk_w_out) { // expect float32
  using executorch::aten::ScalarType;

  fprintf(stderr, "[moe] ENTER %s\n", __func__);
  auto stname = [](ScalarType s) {
    switch (s) {
      case ScalarType::Float:
        return "f32";
      case ScalarType::Half:
        return "f16";
      case ScalarType::BFloat16:
        return "bf16";
      case ScalarType::Long:
        return "i64";
      case ScalarType::Int:
        return "i32";
      default:
        return "other";
    }
  };

  // --- print basic information
  fprintf(
      stderr,
      "[moe] gate args: top_k=%ld, rsf=%.6f, norm=%d, score=%ld, method=%ld, "
      "n_group=%ld, topk_group=%ld, alpha=%.6f\n",
      (long)top_k,
      routed_scaling_factor,
      (int)norm_topk_prob,
      (long)scoring_id,
      (long)method_id,
      (long)n_group,
      (long)topk_group,
      alpha);
  fprintf(
      stderr,
      "[moe] hidden: dim=%ld [%ld,%ld,%ld] dtype=%s\n",
      (long)hidden.dim(),
      (long)hidden.size(0),
      (long)hidden.size(1),
      (long)hidden.size(2),
      stname(hidden.scalar_type()));
  fprintf(
      stderr,
      "[moe] weight: dim=%ld [%ld,%ld] dtype=%s\n",
      (long)weight.dim(),
      (long)weight.size(0),
      (long)weight.size(1),
      stname(weight.scalar_type()));
  fprintf(
      stderr,
      "[moe] out.idx: numel=%ld dtype=%s | out.w: numel=%ld dtype=%s\n",
      (long)topk_idx_out.numel(),
      stname(topk_idx_out.scalar_type()),
      (long)topk_w_out.numel(),
      stname(topk_w_out.scalar_type()));
  if (e_score_correction_bias.numel() > 0) {
    fprintf(
        stderr,
        "[moe] bias: numel=%ld dtype=%s\n",
        (long)e_score_correction_bias.numel(),
        stname(e_score_correction_bias.scalar_type()));
  } else {
    fprintf(stderr, "[moe] bias: <EMPTY>\n");
  }
  fflush(stderr);

  // --- shape check
  if (hidden.dim() != 3 || weight.dim() != 2) {
    fprintf(
        stderr,
        "[moe][ERR] hidden.dim=%ld (expect 3), weight.dim=%ld (expect 2)\n",
        (long)hidden.dim(),
        (long)weight.dim());
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const int64_t B = hidden.size(0), T = hidden.size(1), H = hidden.size(2);
  const int64_t E = weight.size(0);
  if (weight.size(1) != H || top_k <= 0 || top_k > E) {
    fprintf(
        stderr,
        "[moe][ERR] size mismatch: weight.size(1)=%ld vs H=%ld, top_k=%ld, E=%ld\n",
        (long)weight.size(1),
        (long)H,
        (long)top_k,
        (long)E);
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const int64_t need = B * T * top_k;
  if (topk_idx_out.numel() != need || topk_w_out.numel() != need) {
    fprintf(
        stderr,
        "[moe][ERR] out numel mismatch: need=%ld, idx_out=%ld, w_out=%ld\n",
        (long)need,
        (long)topk_idx_out.numel(),
        (long)topk_w_out.numel());
    ctx.fail(Error::InvalidArgument);
    return;
  }

  // --- dtype check (critical!)
  if (hidden.scalar_type() != ScalarType::Float ||
      weight.scalar_type() != ScalarType::Float) {
    fprintf(stderr, "[moe][ERR] hidden/weight must be f32\n");
    ctx.fail(Error::InvalidArgument);
    return;
  }
  if (topk_idx_out.scalar_type() != ScalarType::Long) {
    fprintf(
        stderr,
        "[moe][ERR] topk_idx_out must be int64 (got %s)\n",
        stname(topk_idx_out.scalar_type()));
    ctx.fail(Error::InvalidArgument);
    return;
  }
  if (topk_w_out.scalar_type() != ScalarType::Float) {
    fprintf(
        stderr,
        "[moe][ERR] topk_w_out must be f32 (got %s)\n",
        stname(topk_w_out.scalar_type()));
    ctx.fail(Error::InvalidArgument);
    return;
  }

  // bias: must be f32 when non-empty; method==2(noaux_tc) and bias
  // empty/non-f32 -> error
  const float* bias = nullptr;
  if (e_score_correction_bias.numel() > 0) {
    if (e_score_correction_bias.scalar_type() != ScalarType::Float) {
      fprintf(
          stderr,
          "[moe][ERR] bias must be f32 when provided (got %s)\n",
          stname(e_score_correction_bias.scalar_type()));
      ctx.fail(Error::InvalidArgument);
      return;
    }
    bias = e_score_correction_bias.const_data_ptr<float>();
  } else {
    if (method_id == 2 /*TOPK_NOAUX_TC*/) {
      fprintf(
          stderr,
          "[moe][ERR] method_id==2(noaux_tc) requires non-empty f32 bias\n");
      ctx.fail(Error::InvalidArgument);
      return;
    }
  }

  // --- safe: double -> float
  const float rsf = static_cast<float>(routed_scaling_factor);
  (void)alpha; // placeholder; if your kernel does not need alpha, ignore it

  // --- call kernel
  deepseek::moe_gate_forward(
      /*hidden*/ cf32(hidden),
      /*weight*/ cf32(weight),
      /*bias*/ bias,
      /*B*/ B,
      /*T*/ T,
      /*H*/ H,
      /*E*/ E,
      /*top_k*/ top_k,
      /*score*/ static_cast<int>(scoring_id),
      /*method*/ static_cast<int>(method_id),
      /*n_group*/ n_group,
      /*topk_g*/ topk_group,
      /*norm*/ norm_topk_prob,
      /*scale*/ rsf,
      /*idx*/ i64(topk_idx_out),
      /*w*/ f32(topk_w_out));

  // --- post-check: ensure idx ∈ [0,E-1], weights are non-negative and no
  // NaN/Inf, and normalize if needed
  int64_t* idx_ptr = i64(topk_idx_out);
  float* w_ptr = f32(topk_w_out);
  const int64_t n_tokens = B * T;
  const int64_t k = top_k;

  int64_t min_idx = INT64_MAX, max_idx = INT64_MIN;
  bool idx_fixed = false, w_fixed = false;

  // first count, then clamp idx out of range to [0, E-1]
  for (int64_t i = 0; i < n_tokens * k; ++i) {
    int64_t v = idx_ptr[i];
    if (v < 0) {
      idx_ptr[i] = 0;
      idx_fixed = true;
    } else if (v >= E) {
      idx_ptr[i] = E - 1;
      idx_fixed = true;
    }
    min_idx = std::min<int64_t>(min_idx, idx_ptr[i]);
    max_idx = std::max<int64_t>(max_idx, idx_ptr[i]);
  }

  // clean NaN/Inf/negative values in weights
  auto is_bad = [](float x) {
    return !(x == x) /*NaN*/ || std::isinf(x) || x < 0.0f;
  };
  for (int64_t i = 0; i < n_tokens * k; ++i) {
    if (is_bad(w_ptr[i])) {
      w_ptr[i] = 0.0f;
      w_fixed = true;
    }
  }

  // normalize each token (strictly to rsf when norm_topk_prob==true, sum ≈ rsf)
  if (k > 1 && norm_topk_prob) {
    for (int64_t t = 0; t < n_tokens; ++t) {
      float sum = 0.0f;
      float* row = w_ptr + t * k;
      for (int64_t j = 0; j < k; ++j)
        sum += row[j];
      if (!(sum > 0.0f) || !std::isfinite(sum)) {
        // if all zeros or abnormal, give a uniform distribution
        for (int64_t j = 0; j < k; ++j)
          row[j] = 1.0f / static_cast<float>(k);
        sum = 1.0f;
        w_fixed = true;
      }
      const float scale = static_cast<float>(routed_scaling_factor) / sum;
      for (int64_t j = 0; j < k; ++j)
        row[j] *= scale;
    }
  } else {
    // do not force normalization, but still clip very large values (avoid
    // overflow)
    const float cap = 1e6f;
    for (int64_t i = 0; i < n_tokens * k; ++i) {
      if (w_ptr[i] > cap) {
        w_ptr[i] = cap;
        w_fixed = true;
      }
    }
  }

  fprintf(
      stderr,
      "[moe] gate post-check: idx[min=%ld,max=%ld]%s, w%s\n",
      (long)min_idx,
      (long)max_idx,
      idx_fixed ? " (clamped)" : "",
      w_fixed ? " (cleaned/normalized)" : "");
  fflush(stderr);

  fprintf(stderr, "[moe] EXIT %s ok\n", __func__);
  fflush(stderr);
}

//==================================
// 3) MoE GLU aggregation (single out Tensor)
//==================================
Tensor& moe_forward_glu_out(
    KernelRuntimeContext& ctx,
    const Tensor& hidden, // [B, T, H]
    const Tensor& topk_idx, // [B*T, k] (int64)
    const Tensor& topk_weight, // [B*T, k] (float)
    const TensorList& Wg, // E * [I, H]
    const TensorList& Bg, // E * [I] or empty
    const TensorList& Wu, // E * [I, H]
    const TensorList& Bu, // E * [I] or empty
    const TensorList& Wd, // E * [H, I]
    const TensorList& Bd, // E * [H] or empty
    int64_t act_id,
    const Tensor& act_params, // allow empty
    Tensor& out) { // [B, T, H]

  fprintf(stderr, "[moe] ENTER %s\n", __func__);
  fflush(stderr);

  // Dtype guard check
  if (!is_f32(hidden)) {
    fprintf(
        stderr,
        "[moe] %s expects FP32. hidden=%d\n",
        __func__,
        (int)hidden.scalar_type());
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  if (hidden.dim() != 3) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  const int64_t B = hidden.size(0), T = hidden.size(1), H = hidden.size(2);
  if (out.dim() != 3 || out.size(0) != B || out.size(1) != T ||
      out.size(2) != H) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  if (topk_idx.dim() != 2 || topk_weight.dim() != 2 ||
      topk_idx.size(0) != B * T || topk_weight.size(0) != B * T ||
      topk_idx.size(1) != topk_weight.size(1)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  const int64_t E = static_cast<int64_t>(Wg.size());
  if (!(E == (int64_t)Bg.size() && E == (int64_t)Wu.size() &&
        E == (int64_t)Bu.size() && E == (int64_t)Wd.size() &&
        E == (int64_t)Bd.size())) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  // collect expert weight pointers and intermediate dimension
  std::vector<const float*> pWg(E), pBg(E), pWu(E), pBu(E), pWd(E), pBd(E);
  std::vector<int64_t> I(E);
  for (int64_t e = 0; e < E; ++e) {
    pWg[e] = Wg[e].const_data_ptr<float>();
    pBg[e] = Bg[e].numel() ? Bg[e].const_data_ptr<float>() : nullptr;
    pWu[e] = Wu[e].const_data_ptr<float>();
    pBu[e] = Bu[e].numel() ? Bu[e].const_data_ptr<float>() : nullptr;
    pWd[e] = Wd[e].const_data_ptr<float>();
    pBd[e] = Bd[e].numel() ? Bd[e].const_data_ptr<float>() : nullptr;
    I[e] = Wg[e].size(0); // expert intermediate_size
  }

  const float* ap =
      act_params.numel() ? act_params.const_data_ptr<float>() : nullptr;
  const int nparam =
      act_params.numel() ? static_cast<int>(act_params.numel()) : 0;

  deepseek::moe_forward_glu(
      /*x*/ cf32(hidden),
      /*B*/ B,
      /*T*/ T,
      /*H*/ H,
      /*topk_i*/ ci64(topk_idx),
      /*topk_w*/ cf32(topk_weight),
      /*k*/ topk_weight.size(1),
      /*Wg*/ pWg.data(),
      /*Bg*/ pBg.data(),
      /*Wu*/ pWu.data(),
      /*Bu*/ pBu.data(),
      /*Wd*/ pWd.data(),
      /*Bd*/ pBd.data(),
      /*I*/ I.data(),
      /*act_id*/ static_cast<int>(act_id),
      /*act_p*/ ap,
      /*np*/ nparam,
      /*out*/ f32(out));

  fprintf(stderr, "[moe] EXIT %s ok\n", __func__);
  fflush(stderr);

  return out;
}

// Constructor to confirm library is loaded
extern "C" __attribute__((constructor)) void _et_moe_loaded() {
  fprintf(stderr, "[moe] libdeepseek_moe_execu.so loaded\n");
}

// register: default name & .out name both point to out implementation
EXECUTORCH_LIBRARY(deepseek_moe, "mlp_glu", mlp_glu_out);
EXECUTORCH_LIBRARY(deepseek_moe, "mlp_glu.out", mlp_glu_out);

EXECUTORCH_LIBRARY(deepseek_moe, "moe_gate_forward", moe_gate_forward_out);
EXECUTORCH_LIBRARY(deepseek_moe, "moe_gate_forward.out", moe_gate_forward_out);

EXECUTORCH_LIBRARY(deepseek_moe, "moe_forward_glu", moe_forward_glu_out);
EXECUTORCH_LIBRARY(deepseek_moe, "moe_forward_glu.out", moe_forward_glu_out);
