#include <torch/extension.h>
#include <cmath>
#include <vector>

// ---------------------- Activation mapping (19 kinds) ----------------------
enum Act {
  GELU_ERF = 0, // "gelu" / "gelu_python" (exact erf form)
  GELU_TANH = 1, // "gelu_new" / "gelu_pytorch_tanh" / "gelu_accurate"
  GELU_FAST = 2, // "gelu_fast"
  QUICK_GELU = 3, // "quick_gelu"
  SILU = 4, // "silu" / "swish"
  RELU = 5, // "relu"
  RELU6 = 6, // "relu6"
  RELU_SQUARED = 7, // "relu2"
  TANH = 8, // "tanh"
  SIGMOID = 9, // "sigmoid"
  LINEAR = 10, // "linear"
  MISH = 11, // "mish"
  LEAKY_RELU = 12, // "leaky_relu" (params[0] = negative_slope, default 0.01)
  LAPLACE = 13, // "laplace" (params[0]=mu, params[1]=sigma; default 0.707107,
                // 0.282095)
  CLIPPED_GELU = 14 // "gelu_10" (params[0]=min, params[1]=max; default -10, 10)
};

static inline double
_param(const c10::optional<torch::Tensor>& p, int i, double def) {
  if (!p.has_value())
    return def;
  const auto& t = p.value();
  if (t.numel() <= i)
    return def;
  return t[i].item<double>();
}

static inline torch::Tensor apply_act(
    const torch::Tensor& x,
    int64_t act_id,
    const c10::optional<torch::Tensor>& params) {
  using torch::Tensor;
  switch (act_id) {
    case GELU_ERF: { // x*0.5*(1+erf(x/sqrt(2)))
      return x * 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
    }
    case GELU_TANH: { // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715x^3)))
      const double c = std::sqrt(2.0 / 3.14159265358979323846);
      return 0.5 * x * (1.0 + torch::tanh(c * (x + 0.044715 * x * x * x)));
    }
    case GELU_FAST: { // 0.5*x*(1+tanh(0.7978845608*x*(1+0.044715*x*x)))
      return 0.5 * x *
          (1.0 + torch::tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)));
    }
    case QUICK_GELU: { // x*sigmoid(1.702*x)
      return x * torch::sigmoid(1.702 * x);
    }
    case SILU: { // x*sigmoid(x)
      return x * torch::sigmoid(x);
    }
    case RELU: {
      return torch::relu(x);
    }
    case RELU6: {
      return torch::clamp(x, 0, 6);
    }
    case RELU_SQUARED: { // (relu(x))^2
      auto r = torch::relu(x);
      return r * r;
    }
    case TANH: {
      return torch::tanh(x);
    }
    case SIGMOID: {
      return torch::sigmoid(x);
    }
    case LINEAR: {
      return x;
    }
    case MISH: { // x*tanh(softplus(x))
      return x * torch::tanh(torch::softplus(x));
    }
    case LEAKY_RELU: {
      const double slope = _param(params, 0, 0.01);
      return torch::where(x >= 0, x, x * slope);
    }
    case LAPLACE: { // 0.5*(1+erf((x-mu)/(sigma*sqrt(2))))
      const double mu = _param(params, 0, 0.707107);
      const double sigma = _param(params, 1, 0.282095);
      auto z = (x - mu) / (sigma * std::sqrt(2.0));
      return 0.5 * (1.0 + torch::erf(z));
    }
    case CLIPPED_GELU: { // clip(gelu_exact(x), min, max)
      const double mn = _param(params, 0, -10.0);
      const double mx = _param(params, 1, 10.0);
      auto g = x * 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
      return torch::clamp(g, mn, mx);
    }
    default:
      return x;
  }
}

// GLU expert: down( act(gate(x))*up(x) )
static torch::Tensor mlp_glu_forward(
    const torch::Tensor& x,
    const torch::Tensor& w_gate,
    const torch::Tensor& b_gate,
    const torch::Tensor& w_up,
    const torch::Tensor& b_up,
    const torch::Tensor& w_down,
    const torch::Tensor& b_down,
    int64_t act_id,
    const c10::optional<torch::Tensor>& act_params) {
  auto a = torch::linear(x, w_gate, b_gate);
  auto b = torch::linear(x, w_up, b_up);
  auto h = apply_act(a, act_id, act_params) * b;
  return torch::linear(h, w_down, b_down);
}

// Dense GLU MLP OP: y = down( act(gate(x)) * up(x) )
torch::Tensor mlp_glu(
    torch::Tensor hidden, // (B,T,H) or (N,H)
    torch::Tensor gate_w,
    torch::Tensor gate_b,
    torch::Tensor up_w,
    torch::Tensor up_b,
    torch::Tensor down_w,
    torch::Tensor down_b,
    int64_t act_id,
    c10::optional<torch::Tensor> act_params) {
  bool need_view_back = (hidden.dim() == 3);
  auto x2d = need_view_back ? hidden.reshape({-1, hidden.size(-1)}) : hidden;

  auto a = torch::linear(x2d, gate_w, gate_b);
  auto b = torch::linear(x2d, up_w, up_b);
  auto h = apply_act(a, act_id, act_params) * b;
  auto y2d = torch::linear(h, down_w, down_b);

  return need_view_back ? y2d.view_as(hidden) : y2d;
}

// MoE forward (GLU, supports 19 activations)
torch::Tensor moe_forward_glu(
    torch::Tensor hidden,
    torch::Tensor topk_idx,
    torch::Tensor topk_weight,
    std::vector<torch::Tensor> gate_w,
    std::vector<torch::Tensor> gate_b,
    std::vector<torch::Tensor> up_w,
    std::vector<torch::Tensor> up_b,
    std::vector<torch::Tensor> down_w,
    std::vector<torch::Tensor> down_b,
    int64_t act_id,
    c10::optional<torch::Tensor> act_params) {
  auto flat = hidden.reshape({-1, hidden.size(-1)});
  auto idx = topk_idx.reshape({-1}); // (n*k,)
  auto wgt = topk_weight.reshape({-1, 1}); // (n*k,1)

  auto flat_rep = flat.repeat_interleave(topk_idx.size(-1), 0);
  torch::Tensor y = torch::empty_like(flat_rep);

  for (int i = 0; i < (int)gate_w.size(); ++i) {
    auto mask = (idx == i);
    if (mask.any().item<bool>()) {
      auto in = flat_rep.index({mask});
      y.index_put_(
          {mask},
          mlp_glu_forward(
              in,
              gate_w[i],
              gate_b[i],
              up_w[i],
              up_b[i],
              down_w[i],
              down_b[i],
              act_id,
              act_params));
    }
  }

  // gather & weighted sum over k
  auto view_shape = topk_weight.sizes().vec(); // ..., k
  view_shape.push_back(-1); // ..., k, D
  auto numel = topk_weight.numel(); // n*k
  auto y_weighted = (y.reshape({numel, -1}) * wgt).view(view_shape);
  int64_t sum_dim = static_cast<int64_t>(topk_weight.dim()) - 1; // the k-dim
  y = y_weighted.sum(sum_dim);
  return y.view_as(hidden);
}

// Gate (scores, top-k selection, aux loss)
enum Score { SCORE_SOFTMAX = 0, SCORE_SIGMOID = 1 };
enum Method {
  TOPK_GREEDY = 0,
  TOPK_GROUP_LIMITED_GREEDY = 1,
  TOPK_NOAUX_TC = 2
};

std::tuple<torch::Tensor, torch::Tensor> moe_gate_forward(
    torch::Tensor hidden, // (B,T,H)
    torch::Tensor weight, // (E,H)
    int64_t top_k,
    double routed_scaling_factor,
    bool norm_topk_prob,
    int64_t scoring_id, // 0 softmax, 1 sigmoid
    int64_t method_id, // 0 greedy, 1 group_limited_greedy, 2 noaux_tc
    int64_t n_group, // 0 if unused
    int64_t topk_group, // 0 if unused
    bool training,
    double alpha,
    bool seq_aux,
    c10::optional<torch::Tensor> e_score_correction_bias // for noaux_tc
) {
  TORCH_CHECK(hidden.dim() == 3, "hidden must be (B,T,H)");
  const auto B = hidden.size(0), T = hidden.size(1), H = hidden.size(2);
  const auto E = weight.size(0);
  auto dtype_in = hidden.dtype();

  // logits: (n,H)*(E,H)^T -> (n,E)
  auto flat = hidden.reshape({-1, H}).to(torch::kFloat32);
  auto logits = torch::linear(flat, weight.to(torch::kFloat32), {});
  torch::Tensor scores;
  if (scoring_id == SCORE_SOFTMAX)
    scores = torch::softmax(logits, -1);
  else if (scoring_id == SCORE_SIGMOID)
    scores = torch::sigmoid(logits);
  else
    TORCH_CHECK(false, "unsupported scoring_id");

  // top-k
  torch::Tensor topk_idx, topk_weight;
  if (method_id == TOPK_GREEDY) {
    std::tie(topk_weight, topk_idx) =
        torch::topk(scores, top_k, -1, true, false);
  } else if (method_id == TOPK_GROUP_LIMITED_GREEDY) {
    TORCH_CHECK(n_group > 0 && (E % n_group) == 0, "invalid group settings");
    TORCH_CHECK(topk_group > 0 && topk_group <= n_group, "invalid topk_group");
    const auto per_group = E / n_group;
    auto s3 = scores.view({B * T, n_group, per_group});
    auto group_scores = std::get<0>(s3.max(-1, false));
    auto group_idx =
        std::get<1>(torch::topk(group_scores, topk_group, -1, true, false));
    auto group_mask = torch::zeros_like(group_scores);
    group_mask.scatter_(1, group_idx, 1);
    auto score_mask = group_mask.unsqueeze(-1)
                          .expand({B * T, n_group, per_group})
                          .reshape({B * T, E});
    auto tmp_scores = scores.masked_fill(~score_mask.to(torch::kBool), 0.0);
    std::tie(topk_weight, topk_idx) =
        torch::topk(tmp_scores, top_k, -1, true, false);
  } else if (method_id == TOPK_NOAUX_TC) {
    TORCH_CHECK(!training, "noaux_tc is inference-only");
    TORCH_CHECK(
        n_group > 0 && (E % n_group) == 0 && topk_group > 0 &&
            topk_group <= n_group,
        "invalid group settings (noaux_tc)");
    TORCH_CHECK(
        e_score_correction_bias.has_value(),
        "noaux_tc needs e_score_correction_bias");
    auto bias = e_score_correction_bias.value()
                    .to(scores.dtype())
                    .unsqueeze(0); // (1,E)
    auto scores_for_choice = scores + bias;
    const auto per_group = E / n_group;
    auto s3 = scores_for_choice.view({B * T, n_group, per_group});
    auto top2 = std::get<0>(torch::topk(s3, 2, -1, true, false)); // (n,G,2)
    auto group_scores = top2.sum(-1); // (n,G)
    auto group_idx =
        std::get<1>(torch::topk(group_scores, topk_group, -1, true, false));
    auto group_mask = torch::zeros_like(group_scores);
    group_mask.scatter_(1, group_idx, 1);
    auto score_mask = group_mask.unsqueeze(-1)
                          .expand({B * T, n_group, per_group})
                          .reshape({B * T, E});
    auto tmp_scores =
        scores_for_choice.masked_fill(~score_mask.to(torch::kBool), 0.0);
    topk_idx = std::get<1>(torch::topk(tmp_scores, top_k, -1, true, false));
    topk_weight = scores.gather(1, topk_idx); // weights from original scores
  } else {
    TORCH_CHECK(false, "unsupported topk method");
  }

  // normalize / scale
  if (top_k > 1 && norm_topk_prob) {
    auto denom = topk_weight.sum(1, /*keepdim=*/true) + 1e-20;
    topk_weight = topk_weight / denom * routed_scaling_factor;
  } else {
    topk_weight = topk_weight * routed_scaling_factor;
  }
  topk_weight = topk_weight.to(dtype_in);
  topk_idx = topk_idx.to(torch::kLong);

  // aux loss calculation removed - not needed for inference
  return {topk_idx, topk_weight};
}

// ---- forward declaration ----
static std::tuple<torch::Tensor, torch::Tensor> moe_gate_forward_adapt(
    torch::Tensor,
    torch::Tensor,
    int64_t,
    double,
    bool,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    bool,
    double,
    bool,
    torch::Tensor);

static std::tuple<torch::Tensor&, torch::Tensor&> moe_gate_forward_out_adapt(
    torch::Tensor,
    torch::Tensor,
    int64_t,
    double,
    bool,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    bool,
    double,
    bool,
    torch::Tensor,
    torch::Tensor&,
    torch::Tensor&);

// ---------------------- Register ops ----------------------
TORCH_LIBRARY(deepseek_moe, m) {
  // Dense MLP (general GLU)
  m.def(
      "mlp_glu(Tensor hidden, Tensor gate_w, Tensor gate_b, "
      "Tensor up_w, Tensor up_b, Tensor down_w, Tensor down_b, "
      "int act_id, Tensor? act_params=None) -> Tensor");
  m.impl("mlp_glu", torch::dispatch(torch::kCPU, mlp_glu));

  // MoE forward
  m.def(
      "moe_forward_glu(Tensor hidden, Tensor topk_idx, Tensor topk_weight, "
      "Tensor[] gate_w, Tensor[] gate_b, Tensor[] up_w, Tensor[] up_b, "
      "Tensor[] down_w, Tensor[] down_b, int act_id, Tensor? act_params=None) -> Tensor");
  m.impl("moe_forward_glu", torch::dispatch(torch::kCPU, moe_forward_glu));

  // Gate (no-optional schema; empty tensor == None)
  m.def(
      "moe_gate_forward(Tensor hidden, Tensor weight, int top_k, float routed_scaling_factor, "
      "bool norm_topk_prob, int scoring_id, int method_id, int n_group, int topk_group, "
      "bool training, float alpha, bool seq_aux, Tensor e_score_correction_bias) "
      "-> (Tensor, Tensor)");
  m.impl(
      "moe_gate_forward", torch::dispatch(torch::kCPU, moe_gate_forward_adapt));
}

// ======== OUT VARIANTS (PyTorch-side) ========
static torch::Tensor& mlp_glu_out_kernel(
    torch::Tensor hidden,
    torch::Tensor gate_w,
    torch::Tensor gate_b,
    torch::Tensor up_w,
    torch::Tensor up_b,
    torch::Tensor down_w,
    torch::Tensor down_b,
    int64_t act_id,
    c10::optional<torch::Tensor> act_params,
    torch::Tensor& out) {
  // directly reuse existing implementation and copy_ to out (simple and
  // reliable)
  auto y = mlp_glu(
      hidden, gate_w, gate_b, up_w, up_b, down_w, down_b, act_id, act_params);
  out.copy_(y);
  return out;
}

static std::tuple<torch::Tensor&, torch::Tensor&> moe_gate_forward_out_kernel(
    torch::Tensor hidden,
    torch::Tensor weight,
    int64_t top_k,
    double routed_scaling_factor,
    bool norm_topk_prob,
    int64_t scoring_id,
    int64_t method_id,
    int64_t n_group,
    int64_t topk_group,
    bool training,
    double alpha,
    bool seq_aux,
    c10::optional<torch::Tensor> e_score_correction_bias,
    torch::Tensor& topk_idx_out,
    torch::Tensor& topk_w_out) {
  auto [idx, w] = moe_gate_forward(
      hidden,
      weight,
      top_k,
      routed_scaling_factor,
      norm_topk_prob,
      scoring_id,
      method_id,
      n_group,
      topk_group,
      training,
      alpha,
      seq_aux,
      e_score_correction_bias);
  topk_idx_out.copy_(idx);
  topk_w_out.copy_(w);
  return {topk_idx_out, topk_w_out};
}

static torch::Tensor& moe_forward_glu_out_kernel(
    torch::Tensor hidden,
    torch::Tensor topk_idx,
    torch::Tensor topk_weight,
    std::vector<torch::Tensor> gate_w,
    std::vector<torch::Tensor> gate_b,
    std::vector<torch::Tensor> up_w,
    std::vector<torch::Tensor> up_b,
    std::vector<torch::Tensor> down_w,
    std::vector<torch::Tensor> down_b,
    int64_t act_id,
    c10::optional<torch::Tensor> act_params,
    torch::Tensor& out) {
  auto y = moe_forward_glu(
      hidden,
      topk_idx,
      topk_weight,
      gate_w,
      gate_b,
      up_w,
      up_b,
      down_w,
      down_b,
      act_id,
      act_params);
  out.copy_(y);
  return out;
}

// ---- Adapters: accept Tensor (possibly empty) and forward as optional ----
static std::tuple<torch::Tensor, torch::Tensor> moe_gate_forward_adapt(
    torch::Tensor hidden,
    torch::Tensor weight,
    int64_t top_k,
    double routed_scaling_factor,
    bool norm_topk_prob,
    int64_t scoring_id,
    int64_t method_id,
    int64_t n_group,
    int64_t topk_group,
    bool training,
    double alpha,
    bool seq_aux,
    torch::Tensor e_score_correction_bias) {
  c10::optional<torch::Tensor> bias;
  if (e_score_correction_bias.defined() &&
      e_score_correction_bias.numel() > 0) {
    bias = e_score_correction_bias;
  }
  return moe_gate_forward(
      hidden,
      weight,
      top_k,
      routed_scaling_factor,
      norm_topk_prob,
      scoring_id,
      method_id,
      n_group,
      topk_group,
      training,
      alpha,
      seq_aux,
      bias);
}

static std::tuple<torch::Tensor&, torch::Tensor&> moe_gate_forward_out_adapt(
    torch::Tensor hidden,
    torch::Tensor weight,
    int64_t top_k,
    double routed_scaling_factor,
    bool norm_topk_prob,
    int64_t scoring_id,
    int64_t method_id,
    int64_t topk_group,
    int64_t n_group,
    bool training,
    double alpha,
    bool seq_aux,
    torch::Tensor e_score_correction_bias,
    torch::Tensor& topk_idx_out,
    torch::Tensor& topk_w_out) {
  c10::optional<torch::Tensor> bias;
  if (e_score_correction_bias.defined() &&
      e_score_correction_bias.numel() > 0) {
    bias = e_score_correction_bias;
  }
  auto [idx, w] = moe_gate_forward(
      hidden,
      weight,
      top_k,
      routed_scaling_factor,
      norm_topk_prob,
      scoring_id,
      method_id,
      n_group,
      topk_group,
      training,
      alpha,
      seq_aux,
      bias);
  topk_idx_out.copy_(idx);
  topk_w_out.copy_(w);
  return {topk_idx_out, topk_w_out};
}

// Register .out variants
TORCH_LIBRARY_FRAGMENT(deepseek_moe, m) {
  // mlp_glu.out
  m.def(
      "mlp_glu.out(Tensor hidden, Tensor gate_w, Tensor gate_b, "
      "Tensor up_w, Tensor up_b, Tensor down_w, Tensor down_b, "
      "int act_id, Tensor? act_params=None, *, Tensor(a!) out) -> Tensor(a!)");
  m.impl("mlp_glu.out", torch::dispatch(torch::kCPU, mlp_glu_out_kernel));

  // moe_gate_forward.out (no-optional; empty tensor == None)
  m.def(
      "moe_gate_forward.out(Tensor hidden, Tensor weight, int top_k, float routed_scaling_factor, "
      "bool norm_topk_prob, int scoring_id, int method_id, int n_group, int topk_group, "
      "bool training, float alpha, bool seq_aux, Tensor e_score_correction_bias, "
      "*, Tensor(a!) topk_idx_out, Tensor(b!) topk_w_out) -> (Tensor(a!), Tensor(b!))");
  m.impl(
      "moe_gate_forward.out",
      torch::dispatch(torch::kCPU, moe_gate_forward_out_adapt));

  // moe_forward_glu.out
  m.def(
      "moe_forward_glu.out(Tensor hidden, Tensor topk_idx, Tensor topk_weight, "
      "Tensor[] gate_w, Tensor[] gate_b, Tensor[] up_w, Tensor[] up_b, "
      "Tensor[] down_w, Tensor[] down_b, int act_id, Tensor? act_params=None, "
      "*, Tensor(a!) out) -> Tensor(a!)");
  m.impl(
      "moe_forward_glu.out",
      torch::dispatch(torch::kCPU, moe_forward_glu_out_kernel));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // empty: ops registered via TORCH_LIBRARY
}
