#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

using at::Tensor;

namespace {

inline Tensor matmul_2d(const Tensor& x, const Tensor& wT) {
  // x: [M,H], wT: [H,O]
  return at::mm(x, wT);
}

// -------- moe_infer_glu (functional) --------
Tensor moe_infer_glu_cpu(
    const Tensor& x, // [N,H]
    const Tensor& topk_idx, // [N,K] long
    const Tensor& topk_w, // [N,K]
    const Tensor& gate_wT_3d, // [E,H,I]
    const Tensor& up_wT_3d, // [E,H,I]
    const Tensor& down_wT_3d) { // [E,I,H]
  TORCH_CHECK(x.dim() == 2, "x must be [N,H]");
  TORCH_CHECK(
      topk_idx.dim() == 2 && topk_idx.scalar_type() == at::kLong,
      "topk_idx [N,K] long");
  TORCH_CHECK(topk_w.dim() == 2, "topk_w [N,K]");

  const int64_t N = x.size(0);
  const int64_t H = x.size(1);
  const int64_t K = topk_idx.size(1);
  const auto dtype = x.scalar_type();

  const int64_t E = gate_wT_3d.size(0);
  const int64_t I = gate_wT_3d.size(2);

  TORCH_CHECK(gate_wT_3d.dim() == 3, "gate_wT_3d must be [E,H,I]");
  TORCH_CHECK(up_wT_3d.dim() == 3, "up_wT_3d must be [E,H,I]");
  TORCH_CHECK(down_wT_3d.dim() == 3, "down_wT_3d must be [E,I,H]");
  TORCH_CHECK(gate_wT_3d.size(1) == H, "gate_wT_3d H mismatch");
  TORCH_CHECK(gate_wT_3d.size(2) == I, "gate_wT_3d I mismatch");
  TORCH_CHECK(
      up_wT_3d.size(0) == E && up_wT_3d.size(1) == H && up_wT_3d.size(2) == I,
      "up_wT_3d shape mismatch");
  TORCH_CHECK(
      down_wT_3d.size(0) == E && down_wT_3d.size(1) == I &&
          down_wT_3d.size(2) == H,
      "down_wT_3d shape mismatch");

  Tensor idx_flat = topk_idx.reshape({N * K}); // [NK]
  Tensor w_flat = topk_w.reshape({N * K}).to(dtype); // [NK]

  // histogram per expert
  auto idxp = idx_flat.data_ptr<int64_t>();
  std::vector<int64_t> hist(E, 0);
  const int64_t NK = idx_flat.numel();
  for (int64_t t = 0; t < NK; ++t) {
    int64_t e = idxp[t];
    TORCH_CHECK(0 <= e && e < E, "expert id out of range");
    hist[e] += 1;
  }

  // bounds
  std::vector<int64_t> bounds(E + 1, 0);
  for (int64_t i = 0; i < E; ++i)
    bounds[i + 1] = bounds[i] + hist[i];

  // counting sort by expert id: perm & inv
  Tensor perm = at::empty({NK}, idx_flat.options());
  Tensor inv = at::empty({NK}, idx_flat.options());
  auto perm_p = perm.data_ptr<int64_t>();
  auto inv_p = inv.data_ptr<int64_t>();
  std::vector<int64_t> cur = bounds;
  for (int64_t t = 0; t < NK; ++t) {
    int64_t e = idxp[t];
    int64_t dst = cur[e]++;
    perm_p[dst] = t;
  }
  for (int64_t s = 0; s < NK; ++s)
    inv_p[perm_p[s]] = s;

  // gather tokens
  Tensor tok = at::arange(N, idx_flat.options());
  Tensor tok_rep = tok.repeat_interleave(K); // [NK]
  Tensor tok_sort = tok_rep.index_select(0, perm); // [NK]
  Tensor rep_sort = x.index_select(0, tok_sort); // [NK,H]

  // per expert
  Tensor y_sort = at::zeros_like(rep_sort); // [NK,H]
  for (int64_t e = 0; e < E; ++e) {
    const int64_t s = bounds[e];
    const int64_t t = bounds[e + 1];
    if (s == t)
      continue;
    Tensor seg = rep_sort.narrow(0, s, t - s); // [Me,H]

    // extract expert weights from 3D tensor
    const Tensor WgT = gate_wT_3d[e]; // [H,I]
    const Tensor WuT = up_wT_3d[e]; // [H,I]
    const Tensor WdT = down_wT_3d[e]; // [I,H]

    Tensor g = matmul_2d(seg.to(WgT.scalar_type()), WgT).to(dtype);
    Tensor u = matmul_2d(seg.to(WuT.scalar_type()), WuT).to(dtype);
    g = at::silu(g);
    Tensor h = g * u; // [Me,I]
    Tensor o = matmul_2d(h.to(WdT.scalar_type()), WdT).to(dtype); // [Me,H]
    y_sort.narrow(0, s, t - s).copy_(o);
  }

  // un-sort to (n,k) and reduce over k with weights
  Tensor y_by_nk = y_sort.index_select(0, inv); // [NK,H]
  Tensor y = (y_by_nk * w_flat.unsqueeze(1)).view({N, K, H}).sum(1); // [N,H]
  return y;
}

// -------- moe_infer_glu.out (single output: must have alias) --------
Tensor moe_infer_glu_out_cpu(
    const Tensor& x,
    const Tensor& topk_idx,
    const Tensor& topk_w,
    const Tensor& gate_wT_3d,
    const Tensor& up_wT_3d,
    const Tensor& down_wT_3d,
    Tensor& out) {
  const auto N = x.size(0), H = x.size(1);
  out.resize_({N, H});
  Tensor y =
      moe_infer_glu_cpu(x, topk_idx, topk_w, gate_wT_3d, up_wT_3d, down_wT_3d);
  out.copy_(y);
  return out;
}

std::tuple<Tensor, Tensor> topk_select_out_cpu(
    const Tensor& scores,
    int64_t top_k,
    bool norm_topk_prob,
    double routed_scaling_factor,
    Tensor& topk_idx_out,
    Tensor& topk_w_out) {
  TORCH_CHECK(scores.dim() == 2, "scores must be 2D [N,E]");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(topk_idx_out.scalar_type() == at::kLong, "idx_out must be long");
  TORCH_CHECK(topk_w_out.scalar_type() == at::kFloat, "w_out must be float");
  TORCH_CHECK(top_k > 0, "top_k must be positive");
  const int64_t N = scores.size(0), E = scores.size(1);
  TORCH_CHECK(top_k <= E, "top_k cannot exceed E");

  topk_idx_out.resize_({N, top_k});
  topk_w_out.resize_({N, top_k});

  const float* sp = scores.data_ptr<float>();
  int64_t* ip = topk_idx_out.data_ptr<int64_t>();
  float* wp = topk_w_out.data_ptr<float>();
  const float rsf = static_cast<float>(routed_scaling_factor);

  std::vector<std::pair<float, int64_t>> buf(E);
  for (int64_t n = 0; n < N; ++n) {
    const float* row = sp + n * E;
    for (int64_t e = 0; e < E; ++e)
      buf[e] = {row[e], e};
    std::partial_sort(
        buf.begin(), buf.begin() + top_k, buf.end(), [](auto& a, auto& b) {
          return a.first > b.first;
        });
    float sum = 0.f;
    for (int64_t k = 0; k < top_k; ++k) {
      ip[n * top_k + k] = buf[k].second;
      float weight = std::max(0.0f, buf[k].first);
      wp[n * top_k + k] = weight;
      sum += weight;
    }
    if (top_k > 1 && norm_topk_prob) {
      float scale = (sum > 0.f && std::isfinite(sum))
          ? (rsf / sum)
          : (rsf / static_cast<float>(top_k));
      for (int64_t k = 0; k < top_k; ++k)
        wp[n * top_k + k] *= scale;
    } else {
      for (int64_t k = 0; k < top_k; ++k)
        wp[n * top_k + k] *= rsf;
    }
  }

  return {topk_idx_out, topk_w_out};
}

std::tuple<Tensor, Tensor> topk_select_cpu(
    const Tensor& scores,
    int64_t top_k,
    bool norm_topk_prob,
    double routed_scaling_factor) {
  const int64_t N = scores.size(0);
  const int64_t K = top_k;
  Tensor idx = at::empty({N, K}, scores.options().dtype(at::kLong));
  Tensor w = at::empty({N, K}, scores.options());
  topk_select_out_cpu(
      scores, top_k, norm_topk_prob, routed_scaling_factor, idx, w);
  return {idx, w};
}

} // namespace

// ---- register schema----
TORCH_LIBRARY(deepseek_moe_split, m) {
  m.def(
      "moe_infer_glu(Tensor x, Tensor topk_idx, Tensor topk_w, Tensor gate_wT_3d, Tensor up_wT_3d, Tensor down_wT_3d) -> Tensor");
  m.def(
      "moe_infer_glu.out(Tensor x, Tensor topk_idx, Tensor topk_w, Tensor gate_wT_3d, Tensor up_wT_3d, Tensor down_wT_3d, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "topk_select(Tensor scores, int top_k, bool norm_topk_prob, float routed_scaling_factor) -> (Tensor, Tensor)");
  m.def(
      "topk_select.out(Tensor scores, int top_k, bool norm_topk_prob, float routed_scaling_factor, *, Tensor(a!) topk_idx_out, Tensor(b!) topk_w_out) -> (Tensor(a!), Tensor(b!))");
}

TORCH_LIBRARY_IMPL(deepseek_moe_split, CPU, m) {
  m.impl("moe_infer_glu", moe_infer_glu_cpu);
  m.impl("moe_infer_glu.out", moe_infer_glu_out_cpu);
  m.impl("topk_select", topk_select_cpu);
  m.impl("topk_select.out", topk_select_out_cpu);
}

// optional: empty pybind11 module body (register with TORCH_LIBRARY)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
