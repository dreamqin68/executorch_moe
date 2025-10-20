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
using ScalarType = executorch::aten::ScalarType;

// -----------------------------
// topk_select.out  (write to two outputs; -> ())
// schema (PyTorch side):
// "topk_select.out(Tensor scores, int top_k, bool norm_topk_prob, float
// routed_scaling_factor, *, Tensor(a!) topk_idx_out, Tensor(b!) topk_w_out) ->
// ()"
// -----------------------------
static void topk_select_out(
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
    return;
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
}

// -----------------------------
// moe_infer_glu.out  (write to output; -> ())
// schema (PyTorch side):
// "moe_infer_glu.out(Tensor x, Tensor topk_idx, Tensor topk_w, Tensor[]
// gate_wT, Tensor[] up_wT, Tensor[] down_wT, *, Tensor(a!) out) -> ()"
// -----------------------------

static inline float silu(float z) {
  // z * sigmoid(z)
  return z / (1.0f + std::exp(-z));
}

static Tensor& moe_infer_glu_out(
    KernelRuntimeContext& ctx,
    const Tensor& x, // [N,H] float32
    const Tensor& topk_idx, // [N,K] int64
    const Tensor& topk_w, // [N,K] float32
    const Tensor& gate_wT_3d, // [E,H,I] float32
    const Tensor& up_wT_3d, // [E,H,I] float32
    const Tensor& down_wT_3d, // [E,I,H] float32
    Tensor& out) // [N,H] float32
{
  static bool logged = false;
  if (!logged) {
    logged = true;
    std::fprintf(
        stderr,
        "[moe_infer_glu_out] CALLED. x=(%ld,%ld) topk=(%ld,%ld) out=(%ld,%ld)\n",
        (long)x.size(0),
        (long)x.size(1),
        (long)topk_idx.size(0),
        (long)topk_idx.size(1),
        (long)out.size(0),
        (long)out.size(1));
    std::fprintf(
        stderr,
        "[moe_infer_glu_out] x_strides=[%ld,%ld] out_strides=[%ld,%ld]\n",
        (long)x.strides()[0],
        (long)x.strides()[1],
        (long)out.strides()[0],
        (long)out.strides()[1]);
    std::fprintf(
        stderr,
        "[moe_infer_glu_out] topk_idx_strides=[%ld,%ld] topk_w_strides=[%ld,%ld]\n",
        (long)topk_idx.strides()[0],
        (long)topk_idx.strides()[1],
        (long)topk_w.strides()[0],
        (long)topk_w.strides()[1]);
    std::fflush(stderr);
  }
  if (x.scalar_type() != ScalarType::Float ||
      topk_w.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float ||
      topk_idx.scalar_type() != ScalarType::Long) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  if (x.dim() != 2 || topk_idx.dim() != 2 || topk_w.dim() != 2 ||
      gate_wT_3d.dim() != 3 || up_wT_3d.dim() != 3 || down_wT_3d.dim() != 3 ||
      out.dim() != 2) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  const int64_t N = x.size(0), H = x.size(1);
  const int64_t K = topk_idx.size(1);
  const int64_t E = gate_wT_3d.size(0);
  const int64_t I = gate_wT_3d.size(2);

  if (topk_idx.size(0) != N || topk_w.size(0) != N || topk_w.size(1) != K ||
      out.size(0) != N || out.size(1) != H || up_wT_3d.size(0) != E ||
      up_wT_3d.size(1) != H || up_wT_3d.size(2) != I ||
      down_wT_3d.size(0) != E || down_wT_3d.size(1) != I ||
      down_wT_3d.size(2) != H) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }

  // ptr & strides
  const float* x_ptr = x.const_data_ptr<float>();
  const int64_t* idx_p = topk_idx.const_data_ptr<int64_t>();
  const float* w_ptr = topk_w.const_data_ptr<float>();
  float* out_ptr = out.mutable_data_ptr<float>();

  std::memset(out_ptr, 0, out.nbytes());

  const int64_t xs0 = x.strides()[0], xs1 = x.strides()[1];
  const int64_t os0 = out.strides()[0], os1 = out.strides()[1];
  const int64_t is0 = topk_idx.strides()[0], is1 = topk_idx.strides()[1];
  const int64_t ws0 = topk_w.strides()[0], ws1 = topk_w.strides()[1];

  // Debug: print strides to check if they're non-contiguous
  std::fprintf(
      stderr,
      "[moe_infer_glu_out] x strides: [%ld, %ld], out strides: [%ld, %ld]\n",
      xs0,
      xs1,
      os0,
      os1);
  std::fprintf(
      stderr, "[moe_infer_glu_out] expected contiguous: [%ld, 1]\n", H);
  std::fprintf(
      stderr,
      "[topk] idx strides=[%ld,%ld], w strides=[%ld,%ld]\n",
      is0,
      is1,
      ws0,
      ws1);

  // Print first sample's top-k data
  if (!logged) {
    for (int64_t k = 0; k < K; ++k) {
      auto e0 = *(idx_p + 0 * is0 + k * is1);
      auto a0 = *(w_ptr + 0 * ws0 + k * ws1);
      std::fprintf(
          stderr,
          "[moe] n0 k%ld -> e=%ld a=%g\n",
          (long)k,
          (long)e0,
          (double)a0);
    }
    std::fflush(stderr);
  }

  // 3D weight strides
  const float* Wg = gate_wT_3d.const_data_ptr<float>();
  const float* Wu = up_wT_3d.const_data_ptr<float>();
  const float* Wd = down_wT_3d.const_data_ptr<float>();
  const int64_t wg_e = gate_wT_3d.strides()[0], wg_h = gate_wT_3d.strides()[1],
                wg_i = gate_wT_3d.strides()[2];
  const int64_t wu_e = up_wT_3d.strides()[0], wu_h = up_wT_3d.strides()[1],
                wu_i = up_wT_3d.strides()[2];
  const int64_t wd_e = down_wT_3d.strides()[0], wd_i = down_wT_3d.strides()[1],
                wd_h = down_wT_3d.strides()[2];

  // ---- Work buffers: g/u/h vectors (length I)
  std::vector<float> g(I), u(I), h(I);

  // ---- Main loop: aggregate K experts for each token
  for (int64_t n = 0; n < N; ++n) {
    // out(n, hdim) base address
    float* outrow = out_ptr + n * os0;

    for (int64_t k = 0; k < K; ++k) {
      const int64_t e = *(idx_p + n * is0 + k * is1);
      if (e < 0 || e >= E) {
        // Defense: skip out-of-bounds expert
        continue;
      }
      const float alpha = *(w_ptr + n * ws0 + k * ws1);
      if (alpha == 0.0f) {
        continue;
      }

      // g = silu(x @ Wg[e]) ; u = x @ Wu[e]
      for (int64_t i = 0; i < I; ++i) {
        float accg = 0.f, accu = 0.f;
        for (int64_t hdim = 0; hdim < H; ++hdim) {
          const float xv = *(x_ptr + n * xs0 + hdim * xs1);
          accg += xv * (*(Wg + e * wg_e + hdim * wg_h + i * wg_i));
          accu += xv * (*(Wu + e * wu_e + hdim * wu_h + i * wu_i));
        }
        g[i] = accg / (1.0f + std::exp(-accg)); // silu
        u[i] = accu;
        h[i] = g[i] * u[i];
      }
      // y = h @ Wd[e]^T (Wd[e] 是 [I,H])
      for (int64_t hdim = 0; hdim < H; ++hdim) {
        float acc = 0.f;
        for (int64_t i = 0; i < I; ++i) {
          acc += h[i] * (*(Wd + e * wd_e + i * wd_i + hdim * wd_h));
        }
        *(outrow + hdim * os1) += alpha * acc;
      }
    } // k
  } // n

  return out;
}

// -----------------------------
// Registration CPU kernels
// -----------------------------
EXECUTORCH_LIBRARY(deepseek_moe_split, "topk_select.out", topk_select_out);
EXECUTORCH_LIBRARY(deepseek_moe_split, "moe_infer_glu.out", moe_infer_glu_out);
EXECUTORCH_LIBRARY(deepseek_moe_split, "topk_select", topk_select_out);
EXECUTORCH_LIBRARY(deepseek_moe_split, "moe_infer_glu", moe_infer_glu_out);

// Constructor function to print debug info when .so is loaded
extern "C" __attribute__((constructor)) void _split_moe_loaded() {
  std::fprintf(stderr, "[split_moe] libdeepseek_moe_execu.so loaded\n");
  std::fprintf(
      stderr,
      "[split_moe] debug: topk_select_out=%p, moe_infer_glu_out=%p\n",
      (void*)&topk_select_out,
      (void*)&moe_infer_glu_out);
  std::fprintf(
      stderr,
      "[split_moe] debug: KernelRuntimeContext type_info=%p\n",
      (void*)&typeid(KernelRuntimeContext));
  std::fflush(stderr);
}
