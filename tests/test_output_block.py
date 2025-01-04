from tests.test_utils import load_data_model
import torch


def test_output_block():
    batch, model = load_data_model()

    max_nei = model.backbone.molecular_graph_cfg.max_neighbors
    hidden_size = model.backbone.global_cfg.hidden_size
    x = model.backbone.data_preprocess(batch)

    N = x.node_padding_mask.shape[0]

    edge_readout = torch.randn(
        (N, max_nei, hidden_size * (model.backbone.gnn_cfg.num_layers + 1)),
        device=batch.pos.device,
    )
    node_readout = torch.randn(
        (N, hidden_size * (model.backbone.gnn_cfg.num_layers + 1)),
        device=batch.pos.device,
    )

    emb = model.backbone(batch)
    energy_output = model.output_heads["energy"](edge_readout, emb)
    force_output = model.output_heads["forces"](node_readout, emb)
    assert force_output["energy"].shape == (batch.natoms.shape[0], 1)
    assert energy_output["forces"].shape == (batch.num_nodes, 3)
