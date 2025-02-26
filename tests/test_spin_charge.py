import numpy as np
import torch

from tests.test_utils import (
    load_one_hot,
    load_preprocess_example,
    load_data_model_general,
)
from src.utils.graph_utils import (
    get_potential,
    potential_full_from_pos,
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

    def test_potential_full(self):
        radius_lr = 5.5
        sigma = 1.0
        epsilon = 1e-6
        twopi = 2.0 * np.pi
        max_num_neighbors = 40
        n_nodes = self.batch.num_nodes
        q = torch.zeros(n_nodes, device=self.batch.pos.device)
        q[0] = 1.0
        batch_ind = self.batch.batch
        pos = self.batch.pos

        potential = potential_full_from_pos(
            batch=batch_ind,
            pos=pos,
            q=q,
            sigma=sigma,
            epsilon=epsilon,
            twopi=twopi,
            radius_lr=radius_lr,
            max_num_neighbors=max_num_neighbors,
        )
        potential = list(potential.cpu().numpy())
        assert potential[0] == 0.0, f"Expected 0.0, got {potential[0]}"
        assert np.isclose(
            float(potential[1]), 0.0477, atol=1e-3
        ), f"Expected 0.0477, got {potential[1]}"
        assert potential[-1] == 0.0, f"Expected 0.0, got {potential[-1]}"

    def test_spin(self):
        q = torch.tensor([1.0, -1.0, 1.0], device=self.batch.pos.device)
        edge_index = torch.tensor([[0, 1], [1, 0]], device=self.batch.pos.device)
        pos = self.batch.pos

        layers = [torch.nn.Linear(in_features=1, out_features=20)]
        layers += [torch.nn.Linear(in_features=20, out_features=1)]
        nn_charge = torch.nn.Sequential(*layers)
        # set nn_charge to all ones
        nn_charge[0].weight.data.fill_(1)
        nn_charge[0].bias.data.fill_(0)
        nn_charge[1].weight.data.fill_(1)
        nn_charge[1].bias.data.fill_(0)

        nn_charge.to(self.batch.pos.device)

        energy_spin_raw = heisenberg_potential_full_from_edge_inds(
            pos=pos,
            edge_index=edge_index,
            q=q,
            nn=nn_charge,
        )

        # print(energy_spin_raw)

        # assert all are zero expect for the first
        benchmark = torch.tensor([-27.9526], device=self.batch.pos.device)
        assert torch.allclose(
            input=energy_spin_raw, other=benchmark
        ), f"Expected -27.9526, got {energy_spin_raw[0]}"
