#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/operator_registry.h>

using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

// Simple SiLU activation function: z * sigmoid(z)
static inline float silu(float z) {
  return z / (1.0f + std::exp(-z));
}

// -----------------------------
// topk_select.out  (write to two outputs; -> ())
// schema (PyTorch side):
// "topk_select.out(Tensor scores, int top_k, bool norm_topk_prob, float
// routed_scaling_factor, *, Tensor(a!) topk_idx_out, Tensor(b!) topk_w_out) ->
// (Tensor(a!), Tensor(b!))"
// -----------------------------
static std::tuple<Tensor&, Tensor&> topk_select_out(
    KernelRuntimeContext& /*ctx*/,
    const Tensor& scores, // [N,E], float32
    int64_t top_k,
    bool norm_topk_prob,
    double routed_scaling_factor,
    Tensor& topk_idx_out, // [N,K], int64
    Tensor& topk_w_out // [N,K], float32
) {
  static bool logged = false;
  if (!logged) {
    logged = true;
    std::fprintf(
        stderr,
        "[topk_select_out] CALLED. scores=(%ld,%ld) topk=%ld\n",
        (long)scores.size(0),
        (long)scores.size(1),
        (long)top_k);
    std::fflush(stderr);
  }
  const int64_t N = scores.dim() >= 1 ? scores.size(0) : 0;
  const int64_t E = scores.dim() >= 2 ? scores.size(1) : 0;
  const int64_t K = std::max<int64_t>(0, std::min<int64_t>(top_k, E));
  if (K == 0 || N == 0) {
    if (topk_idx_out.nbytes())
      std::memset(
          topk_idx_out.mutable_data_ptr<void>(), 0, topk_idx_out.nbytes());
    if (topk_w_out.nbytes())
      std::memset(topk_w_out.mutable_data_ptr<void>(), 0, topk_w_out.nbytes());
    return {topk_idx_out, topk_w_out};
  }

  const float* s_ptr = scores.const_data_ptr<float>();
  const int64_t s0 = scores.strides()[0];
  const int64_t s1 = scores.strides()[1];

  int64_t* idx_ptr = topk_idx_out.mutable_data_ptr<int64_t>();
  float* w_ptr = topk_w_out.mutable_data_ptr<float>();
  const int64_t i0 = topk_idx_out.strides()[0], i1 = topk_idx_out.strides()[1];
  const int64_t w0 = topk_w_out.strides()[0], w1 = topk_w_out.strides()[1];

  // temporary buffer
  std::vector<int64_t> sel_idx(K, 0);
  std::vector<float> sel_val(K, 0.f);

  for (int64_t n = 0; n < N; ++n) {
    // 1) select K maximum columns (no duplicates)
    //    still use naive O(K*E) scan
    //    (can also use partial_sort, but naive is enough and equivalent)
    //    after each selection, mark the column
    //    here use a small used array; if E is large, can also replace with
    //    "check selected idx"
    std::vector<char> used(E, 0);
    for (int64_t k = 0; k < K; ++k) {
      int64_t best_j = 0;
      float best_v = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < E; ++j) {
        if (used[j])
          continue;
        float v = *(s_ptr + n * s0 + j * s1);
        if (v > best_v) {
          best_v = v;
          best_j = j;
        }
      }
      used[best_j] = 1;
      sel_idx[k] = best_j;
      sel_val[k] = best_v; // first save "original value", then truncate
                           // negative and normalize
    }

    // 2) consistent with PyTorch side: truncate negative weights, then
    // normalize if needed
    //    route_ops.cpp logic:
    //      weight_k = max(0, value_k)
    //      if (K>1 && norm_topk_prob):
    //          scale = (sum>0? rsf/sum : rsf/K)
    //          weight_k *= scale
    //      else:
    //          weight_k *= rsf
    float rsf = static_cast<float>(routed_scaling_factor);
    float sum_pos = 0.f;
    for (int64_t k = 0; k < K; ++k) {
      sel_val[k] = std::max(0.0f, sel_val[k]);
      sum_pos += sel_val[k];
    }

    float norm_scale = 1.f;
    if (K > 1 && norm_topk_prob) {
      norm_scale = (sum_pos > 0.f && std::isfinite(sum_pos))
          ? (rsf / sum_pos)
          : (rsf / static_cast<float>(K));
    }

    // 3) write back to outputs (with strides)
    for (int64_t k = 0; k < K; ++k) {
      idx_ptr[n * i0 + k * i1] = sel_idx[k];
      float w = sel_val[k];
      w = (K > 1 && norm_topk_prob) ? (w * norm_scale) : (w * rsf);
      w_ptr[n * w0 + k * w1] = w;
    }
  }
  return {topk_idx_out, topk_w_out};
}

// -----------------------------
// expert_glu_packed.out  (write to out; -> Tensor(a!))
// schema (PyTorch side):
// "expert_glu_packed.out(Tensor sorted_tokens, Tensor tokens_per_expert,
//  Tensor[] gate_wT, Tensor[] up_wT, Tensor[] down_wT, *, Tensor(a!) out)
//  -> Tensor(a!)"
static Tensor& expert_glu_packed_out(
    KernelRuntimeContext& ctx,
    const Tensor& sorted_tokens, // [M,H] f32
    const Tensor& tokens_per_expert, // [E]   i64
    const executorch::aten::TensorList& gate_wT, // E * [H,I] f32
    const executorch::aten::TensorList& up_wT, // E * [H,I] f32
    const executorch::aten::TensorList& down_wT, // E * [I,H] f32
    Tensor& out) { // [M,H] f32
  using executorch::aten::ScalarType;

  // ---- basic checks
  if (sorted_tokens.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  if (tokens_per_expert.scalar_type() != ScalarType::Long) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  if (sorted_tokens.dim() != 2 || out.dim() != 2 ||
      tokens_per_expert.dim() != 1) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  const int64_t M = sorted_tokens.size(0);
  const int64_t H = sorted_tokens.size(1);
  const int64_t E = tokens_per_expert.size(0);
  if (out.size(0) != M || out.size(1) != H) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  if (E <= 0 || E != (int64_t)gate_wT.size() || E != (int64_t)up_wT.size() ||
      E != (int64_t)down_wT.size()) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  // shape check & get strides
  auto chk_hw = [&](const Tensor& t, int64_t h, int64_t i) {
    return (
        t.dim() == 2 && t.size(0) == h && t.size(1) == i &&
        t.scalar_type() == ScalarType::Float);
  };

  if (!chk_hw(gate_wT[0], H, gate_wT[0].size(1)) ||
      !chk_hw(up_wT[0], H, up_wT[0].size(1)) ||
      !chk_hw(down_wT[0], down_wT[0].size(0), H)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  const int64_t I = gate_wT[0].size(1);

  for (int64_t e = 0; e < E; ++e) {
    if (!(chk_hw(gate_wT[e], H, I) && chk_hw(up_wT[e], H, I))) {
      ctx.fail(Error::InvalidArgument);
      return out;
    }
    if (!(down_wT[e].dim() == 2 && down_wT[e].size(0) == I &&
          down_wT[e].size(1) == H &&
          down_wT[e].scalar_type() == ScalarType::Float)) {
      ctx.fail(Error::InvalidArgument);
      return out;
    }
  }

  // prefix sum
  const int64_t* cnt = tokens_per_expert.const_data_ptr<int64_t>();
  std::vector<int64_t> bounds(E + 1, 0);
  for (int64_t e = 0; e < E; ++e)
    bounds[e + 1] = bounds[e] + std::max<int64_t>(0, cnt[e]);
  if (bounds.back() != M) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  // base pointers & strides
  const float* x_base = sorted_tokens.const_data_ptr<float>();
  float* y_base = out.mutable_data_ptr<float>();
  const int64_t xs0 = sorted_tokens.strides()[0];
  const int64_t xs1 = sorted_tokens.strides()[1];
  const int64_t ys0 = out.strides()[0];
  const int64_t ys1 = out.strides()[1];

  // zero out
  std::memset(y_base, 0, out.nbytes());

  // temporary vector
  std::vector<float> g(I), u(I), h(I);

  // debug: print strides once
  static bool once = false;
  if (!once) {
    once = true;
    std::fprintf(
        stderr,
        "[dbg] xs0=%ld xs1=%ld ys0=%ld ys1=%ld wg0=%ld wg1=%ld wu0=%ld wu1=%ld wd0=%ld wd1=%ld\n",
        (long)xs0,
        (long)xs1,
        (long)ys0,
        (long)ys1,
        (long)gate_wT[0].strides()[0],
        (long)gate_wT[0].strides()[1],
        (long)up_wT[0].strides()[0],
        (long)up_wT[0].strides()[1],
        (long)down_wT[0].strides()[0],
        (long)down_wT[0].strides()[1]);
    std::fflush(stderr);
  }

  for (int64_t e = 0; e < E; ++e) {
    const float* Wg = gate_wT[e].const_data_ptr<float>();
    const float* Wu = up_wT[e].const_data_ptr<float>();
    const float* Wd = down_wT[e].const_data_ptr<float>();

    const int64_t wg0 = gate_wT[e].strides()[0],
                  wg1 = gate_wT[e].strides()[1]; // [H,I]
    const int64_t wu0 = up_wT[e].strides()[0],
                  wu1 = up_wT[e].strides()[1]; // [H,I]
    const int64_t wd0 = down_wT[e].strides()[0],
                  wd1 = down_wT[e].strides()[1]; // [I,H]

    const int64_t s = bounds[e], t = bounds[e + 1];
    for (int64_t m = s; m < t; ++m) {
      const float* xrow = x_base + m * xs0;
      float* yrow = y_base + m * ys0;

      // g = silu(x @ WgT), u = x @ WuT
      for (int64_t i = 0; i < I; ++i) {
        const float* wg_col = Wg + i * wg1; // 取第 i 列
        const float* wu_col = Wu + i * wu1;
        float acc_g = 0.f, acc_u = 0.f;
        for (int64_t hdim = 0; hdim < H; ++hdim) {
          const float xv = *(xrow + hdim * xs1);
          acc_g += xv * *(wg_col + hdim * wg0);
          acc_u += xv * *(wu_col + hdim * wu0);
        }
        g[i] = silu(acc_g);
        u[i] = acc_u;
        h[i] = g[i] * u[i];
      }

      // y = h @ WdT（[I,H]）
      for (int64_t hdim = 0; hdim < H; ++hdim) {
        const float* wd_col = Wd + hdim * wd1; // 第 hdim 列
        float acc = 0.f;
        for (int64_t i = 0; i < I; ++i) {
          acc += h[i] * *(wd_col + i * wd0);
        }
        *(yrow + hdim * ys1) = acc;
      }
    }
  }
  return out;
}

// -----------------------------
// group_by_expert.out (write to three outputs; -> ())
// schema (PyTorch side):
// "group_by_expert.out(Tensor topk_idx, int num_experts, *, Tensor indices_out,
// Tensor inverse_out, Tensor tokens_per_expert_out) -> ()"
// -----------------------------
static void group_by_expert_out(
    KernelRuntimeContext& ctx,
    const Tensor& topk_idx, // [N,K], int64
    int64_t num_experts,
    Tensor& indices_out, // [N*K], int64
    Tensor& inverse_out, // [N*K], int64
    Tensor& tokens_per_expert // [E],   int64
) {
  using executorch::aten::ScalarType;
  if (topk_idx.scalar_type() != ScalarType::Long || topk_idx.dim() != 2) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const int64_t N = topk_idx.size(0), K = topk_idx.size(1);
  const int64_t M = N * K;
  if (num_experts <= 0) {
    ctx.fail(Error::InvalidArgument);
    return;
  }

  if (indices_out.scalar_type() != ScalarType::Long ||
      inverse_out.scalar_type() != ScalarType::Long ||
      tokens_per_expert.scalar_type() != ScalarType::Long) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  if (indices_out.dim() != 1 || inverse_out.dim() != 1 ||
      tokens_per_expert.dim() != 1) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  if (indices_out.size(0) != M || inverse_out.size(0) != M ||
      tokens_per_expert.size(0) != num_experts) {
    ctx.fail(Error::InvalidArgument);
    return;
  }

  const int64_t* p = topk_idx.const_data_ptr<int64_t>();
  int64_t* idx_out = indices_out.mutable_data_ptr<int64_t>();
  int64_t* inv_out = inverse_out.mutable_data_ptr<int64_t>();
  int64_t* cnt = tokens_per_expert.mutable_data_ptr<int64_t>();

  // zero out count
  std::memset(cnt, 0, sizeof(int64_t) * num_experts);

  // 1) count
  for (int64_t m = 0; m < M; ++m) {
    const int64_t e = p[m];
    if (e < 0 || e >= num_experts) {
      ctx.fail(Error::InvalidArgument);
      return;
    }
    cnt[e] += 1;
  }
  // 2) prefix sum
  std::vector<int64_t> off(num_experts + 1, 0);
  for (int64_t e = 0; e < num_experts; ++e)
    off[e + 1] = off[e] + cnt[e];

  // 3) stable fill
  std::vector<int64_t> cursor(num_experts, 0);
  for (int64_t m = 0; m < M; ++m) {
    const int64_t e = p[m];
    const int64_t pos = off[e] + cursor[e]++;
    idx_out[pos] = m;
  }
  // 4) inverse permutation
  for (int64_t pos = 0; pos < M; ++pos) {
    inv_out[idx_out[pos]] = pos;
  }
}

// Register CPU kernels
EXECUTORCH_LIBRARY(deepseek_moe_split, "topk_select.out", topk_select_out);
EXECUTORCH_LIBRARY(
    deepseek_moe_split,
    "expert_glu_packed.out",
    expert_glu_packed_out);
EXECUTORCH_LIBRARY(
    deepseek_moe_split,
    "group_by_expert.out",
    group_by_expert_out);

// Constructor function to print debug info when .so is loaded
extern "C" __attribute__((constructor)) void _split_moe_loaded() {
  std::fprintf(stderr, "[split_moe] libdeepseek_moe_execu.so loaded\n");
  std::fprintf(
      stderr,
      "[split_moe] debug: topk_select_out=%p, expert_glu_packed_out=%p, group_by_expert_out=%p\n",
      (void*)&topk_select_out,
      (void*)&expert_glu_packed_out,
      (void*)&group_by_expert_out);
  std::fprintf(
      stderr,
      "[split_moe] debug: KernelRuntimeContext type_info=%p\n",
      (void*)&typeid(KernelRuntimeContext));
  std::fflush(stderr);
}
