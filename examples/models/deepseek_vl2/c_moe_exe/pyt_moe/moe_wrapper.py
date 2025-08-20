import importlib
import torch
import torch.nn as nn

importlib.import_module("deepseek_moe")

_SCORING = {"softmax": 0, "sigmoid": 1}
_METHOD = {"greedy": 0, "group_limited_greedy": 1, "noaux_tc": 2}

# hidden_act name -> act_id
_ACT_ID = {
    "gelu": 0,
    "gelu_python": 0,
    "gelu_new": 1,
    "gelu_pytorch_tanh": 1,
    "gelu_accurate": 1,
    "gelu_fast": 2,
    "quick_gelu": 3,
    "silu": 4,
    "swish": 4,
    "relu": 5,
    "relu6": 6,
    "relu2": 7,
    "tanh": 8,
    "sigmoid": 9,
    "linear": 10,
    "mish": 11,
    "leaky_relu": 12,
    "laplace": 13,
    "gelu_10": 14,
}


def _bias_or_zeros(lin: nn.Linear) -> torch.Tensor:
    b = getattr(lin, "bias", None)
    if b is not None:
        return b
    return torch.zeros(
        lin.weight.size(0), dtype=lin.weight.dtype, device=lin.weight.device
    )


class DeepseekV2MLP_CPP(nn.Module):
    def __init__(self, mlp_py: nn.Module):
        super().__init__()
        self.gate_proj = mlp_py.gate_proj
        self.up_proj = mlp_py.up_proj
        self.down_proj = mlp_py.down_proj

        name = mlp_py.config.hidden_act
        key = name.lower() if isinstance(name, str) else name
        if key not in _ACT_ID:
            raise ValueError(f"Unsupported hidden_act for dense MLP: {name}")
        self.act_id = int(_ACT_ID[key])
        self.act_params = None

    def forward(self, x: torch.Tensor):
        act_params = self.act_params if self.act_params is not None else x.new_empty(0)
        return torch.ops.deepseek_moe.mlp_glu(
            x,
            self.gate_proj.weight,
            _bias_or_zeros(self.gate_proj),
            self.up_proj.weight,
            _bias_or_zeros(self.up_proj),
            self.down_proj.weight,
            _bias_or_zeros(self.down_proj),
            int(self.act_id),
            act_params,
        )


class DeepseekV2Gate_CPP(nn.Module):
    def __init__(self, gate_py):
        super().__init__()
        self.config = gate_py.config
        # share parameters with python gate (state_dict-compatible)
        self.weight = gate_py.weight
        self.e_score_correction_bias = getattr(gate_py, "e_score_correction_bias", None)

        self.top_k = gate_py.top_k
        self.routed_scaling_factor = gate_py.routed_scaling_factor
        self.norm_topk_prob = gate_py.norm_topk_prob
        self.scoring_id = _SCORING[gate_py.scoring_func]
        self.method_id = _METHOD[gate_py.topk_method]
        self.n_group = gate_py.n_group or 0
        self.topk_group = gate_py.topk_group or 0
        self.alpha = gate_py.alpha
        self.seq_aux = gate_py.seq_aux

    def forward(self, hidden_states: torch.Tensor):
        bias = (
            self.e_score_correction_bias
            if self.e_score_correction_bias is not None
            else hidden_states.new_empty(0, dtype=hidden_states.dtype)
        )
        topk_idx, topk_w = torch.ops.deepseek_moe.moe_gate_forward(
            hidden_states,
            self.weight,
            int(self.top_k),
            float(self.routed_scaling_factor),
            bool(self.norm_topk_prob),
            int(self.scoring_id),
            int(self.method_id),
            int(self.n_group),
            int(self.topk_group),
            bool(self.training),
            float(self.alpha),
            bool(self.seq_aux),
            bias,
        )
        return topk_idx, topk_w, None


class DeepseekV2MoE_CPP(nn.Module):
    def __init__(self, moe_py):
        super().__init__()
        self.gate = DeepseekV2Gate_CPP(moe_py.gate)
        self.experts = moe_py.experts
        self.num_experts_per_tok = moe_py.num_experts_per_tok

        self.n_shared_experts = moe_py.config.n_shared_experts
        if self.n_shared_experts is not None:
            self.shared_experts_cpp = DeepseekV2MLP_CPP(moe_py.shared_experts).eval()

        name = moe_py.config.hidden_act
        if isinstance(name, str):
            key = name.lower()
        else:
            raise ValueError("hidden_act must be a string to map to C++ activations.")
        if key not in _ACT_ID:
            raise ValueError(f"Unsupported hidden_act: {name}")
        self.act_id = int(_ACT_ID[key])
        self.act_params = None

    def forward(self, hidden_states: torch.Tensor):
        topk_idx, topk_w, _ = self.gate(hidden_states)

        gate_w = [e.gate_proj.weight for e in self.experts]
        gate_b = [_bias_or_zeros(e.gate_proj) for e in self.experts]
        up_w = [e.up_proj.weight for e in self.experts]
        up_b = [_bias_or_zeros(e.up_proj) for e in self.experts]
        down_w = [e.down_proj.weight for e in self.experts]
        down_b = [_bias_or_zeros(e.down_proj) for e in self.experts]

        act_params = (
            self.act_params
            if self.act_params is not None
            else hidden_states.new_empty(0)
        )
        out = torch.ops.deepseek_moe.moe_forward_glu(
            hidden_states,
            topk_idx,
            topk_w,
            gate_w,
            gate_b,
            up_w,
            up_b,
            down_w,
            down_b,
            int(self.act_id),
            act_params,
        )
        if self.n_shared_experts is not None:
            out = out + self.shared_experts_cpp(hidden_states)
        return out


def replace_layer_mlp_with_cpp(layer):
    m = layer.mlp
    is_moe = (
        hasattr(m, "experts")
        and hasattr(m, "gate")
        and hasattr(m, "num_experts_per_tok")
    )
    layer.mlp = DeepseekV2MoE_CPP(m).eval() if is_moe else DeepseekV2MLP_CPP(m).eval()
    return layer
