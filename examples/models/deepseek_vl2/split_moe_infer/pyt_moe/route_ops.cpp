#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

using at::Tensor;

namespace {

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

// === expert_glu_packed: Process sorted_tokens by expert according to
// tokens_per_expert and do GLU, then concatenate segments back ===
Tensor expert_glu_packed_cpu(
    const Tensor& sorted_tokens, // [M,H], float
    const Tensor& tokens_per_expert, // [E],  int64
    const std::vector<Tensor>& gate_wT, // E * [H,I], float
    const std::vector<Tensor>& up_wT, // E * [H,I], float
    const std::vector<Tensor>& down_wT) { // E * [I,H], float
  TORCH_CHECK(sorted_tokens.dim() == 2, "sorted_tokens must be [M,H]");
  TORCH_CHECK(
      tokens_per_expert.dim() == 1 &&
          tokens_per_expert.scalar_type() == at::kLong,
      "tokens_per_expert must be int64 1D");
  const auto M = sorted_tokens.size(0);
  const auto H = sorted_tokens.size(1);
  const int64_t E = (int64_t)gate_wT.size();
  TORCH_CHECK(
      (int64_t)up_wT.size() == E && (int64_t)down_wT.size() == E,
      "weight lists len mismatch");
  TORCH_CHECK(
      tokens_per_expert.numel() == E,
      "len(tokens_per_expert) must equal #experts");

  // prefix-sum to calculate [s,t) for each expert segment
  auto cnt_ptr = tokens_per_expert.data_ptr<int64_t>();
  std::vector<int64_t> bounds(E + 1, 0);
  for (int64_t e = 0; e < E; ++e)
    bounds[e + 1] = bounds[e] + std::max<int64_t>(0, cnt_ptr[e]);
  TORCH_CHECK(bounds.back() == M, "sum(tokens_per_expert) must equal M");

  const auto dtype = sorted_tokens.scalar_type();
  Tensor out = at::zeros_like(sorted_tokens); // [M,H]
  for (int64_t e = 0; e < E; ++e) {
    int64_t s = bounds[e], t = bounds[e + 1];
    if (s == t)
      continue;
    const Tensor& WgT = gate_wT[e]; // [H,I]
    const Tensor& WuT = up_wT[e]; // [H,I]
    const Tensor& WdT = down_wT[e]; // [I,H]
    TORCH_CHECK(
        WgT.dim() == 2 && WuT.dim() == 2 && WdT.dim() == 2,
        "weights must be 2D");
    const int64_t I = WgT.size(1);
    TORCH_CHECK(
        WgT.size(0) == H && WuT.size(0) == H && WdT.size(0) == I &&
            WdT.size(1) == H,
        "weight shape mismatch");

    Tensor seg = sorted_tokens.narrow(0, s, t - s); // [Me,H]
    Tensor g = at::mm(seg.to(WgT.scalar_type()), WgT).to(dtype); // [Me,I]
    Tensor u = at::mm(seg.to(WuT.scalar_type()), WuT).to(dtype); // [Me,I]
    g = at::silu(g);
    Tensor h = g * u; // [Me,I]
    Tensor o = at::mm(h.to(WdT.scalar_type()), WdT).to(dtype); // [Me,H]
    out.narrow(0, s, t - s).copy_(o);
  }
  return out;
}

Tensor expert_glu_packed_out_cpu(
    const Tensor& sorted_tokens,
    const Tensor& tokens_per_expert,
    const std::vector<Tensor>& gate_wT,
    const std::vector<Tensor>& up_wT,
    const std::vector<Tensor>& down_wT,
    Tensor& out) {
  out.resize_({sorted_tokens.size(0), sorted_tokens.size(1)});
  Tensor y = expert_glu_packed_cpu(
      sorted_tokens, tokens_per_expert, gate_wT, up_wT, down_wT);
  out.copy_(y);
  return out;
}

// === group_by_expert: Stable grouping of tokens by expert ID ===
std::tuple<Tensor, Tensor, Tensor> group_by_expert_cpu(
    const Tensor& topk_idx, // [N,K], int64
    int64_t num_experts) {
  TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must be [N,K]");
  TORCH_CHECK(topk_idx.scalar_type() == at::kLong, "topk_idx must be int64");
  const int64_t N = topk_idx.size(0), K = topk_idx.size(1);
  const int64_t M = N * K;
  TORCH_CHECK(num_experts > 0, "num_experts must be > 0");
  auto opts_i64 = topk_idx.options().dtype(at::kLong);

  Tensor idxs = at::empty({M}, opts_i64); // permutation
  Tensor inv = at::empty({M}, opts_i64); // reverse permutation
  Tensor cnt = at::zeros({num_experts}, opts_i64); // count

  const int64_t* p = topk_idx.data_ptr<int64_t>();
  int64_t* c = cnt.data_ptr<int64_t>();

  // 1) count
  for (int64_t m = 0; m < M; ++m) {
    const auto e = p[m];
    TORCH_CHECK(0 <= e && e < num_experts, "expert id OOB");
    c[e] += 1;
  }
  // 2) prefix sum -> each expert's starting offset
  std::vector<int64_t> off(num_experts + 1, 0);
  for (int64_t e = 0; e < num_experts; ++e)
    off[e + 1] = off[e] + c[e];

  // 3) stable fill
  std::vector<int64_t> cursor(num_experts, 0);
  int64_t* idx_out = idxs.data_ptr<int64_t>();
  for (int64_t m = 0; m < M; ++m) {
    const auto e = p[m];
    const int64_t pos = off[e] + cursor[e]++;
    idx_out[pos] = m; // m = n*K + k
  }
  // 4) reverse permutation
  int64_t* inv_out = inv.data_ptr<int64_t>();
  for (int64_t pos = 0; pos < M; ++pos) {
    inv_out[idx_out[pos]] = pos;
  }
  return std::make_tuple(idxs, inv, cnt);
}

std::tuple<Tensor, Tensor, Tensor> group_by_expert_out_cpu(
    const Tensor& topk_idx,
    int64_t num_experts,
    Tensor& indices_out,
    Tensor& inverse_out,
    Tensor& tokens_per_expert_out) {
  TORCH_CHECK(
      topk_idx.dim() == 2 && topk_idx.scalar_type() == at::kLong,
      "topk_idx must be [N,K] int64");
  const int64_t N = topk_idx.size(0), K = topk_idx.size(1);
  const int64_t M = N * K;
  auto opts_i64 = topk_idx.options().dtype(at::kLong);
  indices_out.resize_({M});
  inverse_out.resize_({M});
  tokens_per_expert_out.resize_({num_experts});

  Tensor idxs, inv, cnt;
  std::tie(idxs, inv, cnt) = group_by_expert_cpu(topk_idx, num_experts);
  indices_out.copy_(idxs);
  inverse_out.copy_(inv);
  tokens_per_expert_out.copy_(cnt);
  return {indices_out, inverse_out, tokens_per_expert_out};
}

} // namespace

// ---- register schema----
TORCH_LIBRARY(deepseek_moe_split, m) {
  m.def(
      "expert_glu_packed(Tensor sorted_tokens, Tensor tokens_per_expert, Tensor[] gate_wT, Tensor[] up_wT, Tensor[] down_wT) -> Tensor");
  m.def(
      "expert_glu_packed.out(Tensor sorted_tokens, Tensor tokens_per_expert, Tensor[] gate_wT, Tensor[] up_wT, Tensor[] down_wT, *, Tensor(a!) out) -> Tensor(a!)");

  m.def(
      "topk_select(Tensor scores, int top_k, bool norm_topk_prob, float routed_scaling_factor) -> (Tensor, Tensor)");
  m.def(
      "topk_select.out(Tensor scores, int top_k, bool norm_topk_prob, float routed_scaling_factor, *, Tensor(a!) topk_idx_out, Tensor(b!) topk_w_out) -> (Tensor(a!), Tensor(b!))");

  m.def(
      "group_by_expert(Tensor topk_idx, int num_experts) -> (Tensor, Tensor, Tensor)");
  m.def(
      "group_by_expert.out(Tensor topk_idx, int num_experts, *, Tensor(a!) indices_out, Tensor(b!) inverse_out, Tensor(c!) tokens_per_expert_out) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
}

TORCH_LIBRARY_IMPL(deepseek_moe_split, CPU, m) {
  m.impl("expert_glu_packed", expert_glu_packed_cpu);
  m.impl("expert_glu_packed.out", expert_glu_packed_out_cpu);
  m.impl("topk_select", topk_select_cpu);
  m.impl("topk_select.out", topk_select_out_cpu);
  m.impl("group_by_expert", group_by_expert_cpu);
  m.impl("group_by_expert.out", group_by_expert_out_cpu);
}

// optional: empty pybind11 module body (register with TORCH_LIBRARY)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
