import torch

from tests.test_utils import (
    load_one_hot,
    load_preprocess_example,
    load_data_model_general,
)
from src.utils.graph_utils import (
    get_potential,
    potential_full_from_edge_inds,
    heisenberg_potential_full_from_edge_inds,
)


class TestSpinCharge:
    batch, model = load_data_model_general()

    def test_one_hot(self):
        one_hot_data = load_one_hot()
        assert one_hot_data.shape == (49, 5)
        assert one_hot_data.sum() == 49
        assert one_hot_data[0][0] == 1
        assert one_hot_data[20][1] == 1
        assert one_hot_data[-1][2] == 0

    def preprocess_data(self):
        new_data = load_preprocess_example()
        shape_out = new_data.node_direction_expansion.shape
        assert shape_out == (450, 16), f"Expected shape (450, 16), got {shape_out}"

    def test_potential(self):
        q = torch.tensor([1.0, 0.0, 1.0])
        edge_distance = torch.tensor([1.0, 1.0])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        mask_1 = torch.tensor([1])
        mask_2 = torch.tensor([0])
        ind_interactions_1 = torch.tensor([1])
        ind_interactions_2 = torch.tensor([1])

        assert (
            get_potential(q, edge_distance, edge_index, mask_1, ind_interactions_1)
            == torch.tensor(0.0)
        ), f"Expected 0.0, got {get_potential(q, edge_distance, edge_index, mask_1, ind_interactions_1)}"
        assert (
            get_potential(q, edge_distance, edge_index, mask_2, ind_interactions_2)
            == torch.tensor(1.0)
        ), f"Expected 1.0, got {get_potential(q, edge_distance, edge_index, mask_2, ind_interactions_2)}"

    def test_charge(self):
        q = torch.tensor([1.0, -1.0, 1.0], device=self.batch.pos.device)
        edge_index = torch.tensor([[0, 1], [1, 0]], device=self.batch.pos.device)
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            device=self.batch.pos.device,
        )

        energy_spin_raw = potential_full_from_edge_inds(
            pos=pos,
            edge_index=edge_index,
            q=q,
        )

        # assert all are zero expect for the first
        benchmark = torch.tensor([-0.05432664975523949], device=self.batch.pos.device)
        assert torch.allclose(
            input=energy_spin_raw[0], other=benchmark, atol=1e-1
        ), f"Expected -0.05432664975523949, got {energy_spin_raw[0]} "

    def test_spin(self):
        q = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]], device=self.batch.pos.device
        )
        edge_index = torch.tensor([[0, 1], [1, 0]], device=self.batch.pos.device)
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            device=self.batch.pos.device,
        )

        layers = [torch.nn.Linear(in_features=1, out_features=20)]
        layers += [torch.nn.Linear(in_features=20, out_features=1)]
        nn_coupling = torch.nn.Sequential(*layers)
        # set nn_charge to all ones
        nn_coupling[0].weight.data.fill_(1)
        nn_coupling[0].bias.data.fill_(0)
        nn_coupling[1].weight.data.fill_(1)
        nn_coupling[1].bias.data.fill_(0)

        nn_coupling.to(self.batch.pos.device)

        energy_spin_raw = heisenberg_potential_full_from_edge_inds(
            pos=pos,
            edge_index=edge_index,
            q=q,
            nn=nn_coupling,
        )

        # assert all are zero expect for the first
        benchmark = torch.tensor([13.653789520263672], device=self.batch.pos.device)
        assert torch.allclose(
            input=energy_spin_raw[0], other=benchmark, atol=1e-1
        ), f"Expected 13.653789520263672, got {energy_spin_raw[0]} "
