import torch
import torch.nn.functional as F

from ramp_moe.baselines.camera import camera_microexpert_importance


def test_camera_importance_matches_router_conditioned_energy_formula() -> None:
    hidden_states = torch.tensor([[4.0, 0.0], [0.0, 3.0]])
    router_weight = torch.eye(2)
    gate_weights = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.0]],
            [[0.0, 1.0], [0.0, 0.5]],
        ]
    )
    up_weights = torch.tensor(
        [
            [[0.25, 0.0], [0.75, 0.0]],
            [[0.0, 0.25], [0.0, 0.75]],
        ]
    )
    down_weights = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 3.0]],
            [[4.0, 0.0], [0.0, 5.0]],
        ]
    )
    alpha = 0.3

    actual = camera_microexpert_importance(
        hidden_states,
        router_weight,
        gate_weights,
        up_weights,
        down_weights,
        top_k=1,
        alpha=alpha,
        token_chunk_size=1,
    )

    expected = []
    for expert_index, token_index in enumerate((0, 1)):
        token = hidden_states[token_index]
        activation = F.silu(gate_weights[expert_index] @ token)
        activation *= up_weights[expert_index] @ token
        squared = activation.square()
        energy = ((1 - alpha) * squared + alpha * squared)
        energy *= down_weights[expert_index].square().sum(dim=0)
        expected.append(energy)

    torch.testing.assert_close(actual, torch.stack(expected))


def test_camera_importance_rejects_invalid_alpha() -> None:
    hidden_states = torch.ones(1, 2)
    router_weight = torch.ones(2, 2)
    gate_weights = torch.ones(2, 1, 2)
    up_weights = torch.ones(2, 1, 2)
    down_weights = torch.ones(2, 2, 1)

    try:
        camera_microexpert_importance(
            hidden_states,
            router_weight,
            gate_weights,
            up_weights,
            down_weights,
            top_k=1,
            alpha=1.1,
        )
    except ValueError as error:
        assert "alpha" in str(error)
    else:
        raise AssertionError("alpha outside [0, 1] must raise ValueError")
