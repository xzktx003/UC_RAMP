import torch

from ramp_moe.taylor import differentiable_qwen3_channel_gates


class _Expert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 2, bias=False)
        self.up_proj = torch.nn.Linear(2, 2, bias=False)
        self.down_proj = torch.nn.Linear(2, 2, bias=False)
        self.act_fn = torch.nn.SiLU()
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.eye(2))
            self.up_proj.weight.copy_(torch.eye(2))
            self.down_proj.weight.copy_(torch.eye(2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.gate_proj(values)) * self.up_proj(values)
        return self.down_proj(hidden)


class _Moe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList([_Expert()])


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Moe()


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()


def test_differentiable_channel_gate_matches_binary_masking() -> None:
    model = _Model()
    gates = torch.nn.Parameter(torch.tensor([[[1.0, 0.0]]]))
    values = torch.tensor([[1.0, 2.0]])
    original = model.model.layers[0].mlp.experts[0](values)

    with differentiable_qwen3_channel_gates(model, gates) as controller:
        masked = model.model.layers[0].mlp.experts[0](values)
        controller.use_dense = True
        dense = model.model.layers[0].mlp.experts[0](values)

    assert masked[0, 1].item() == 0
    torch.testing.assert_close(masked[0, 0], original[0, 0])
    torch.testing.assert_close(dense, original)


def test_pruned_channel_receives_a_restoration_gradient() -> None:
    model = _Model()
    gates = torch.nn.Parameter(torch.tensor([[[1.0, 0.0]]]))
    values = torch.tensor([[1.0, 2.0]])

    with differentiable_qwen3_channel_gates(model, gates):
        model.model.layers[0].mlp.experts[0](values).sum().backward()

    assert gates.grad is not None
    assert gates.grad[0, 0, 1].item() > 0


def test_differentiable_channel_gate_restores_expert_forward() -> None:
    model = _Model()
    expert = model.model.layers[0].mlp.experts[0]
    gates = torch.nn.Parameter(torch.ones(1, 1, 2))
    original_function = expert.forward.__func__

    with differentiable_qwen3_channel_gates(model, gates):
        assert expert.forward.__func__ is not original_function

    assert expert.forward.__func__ is original_function
