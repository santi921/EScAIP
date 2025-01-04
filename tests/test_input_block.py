from tests.test_utils import load_data_model


def test_input_block():
    batch, model = load_data_model()

    max_nei = model.backbone.molecular_graph_cfg.max_neighbors
    hidden_size = model.backbone.global_cfg.hidden_size

    x = model.backbone.data_preprocess(batch)

    output = model.backbone.input_block(x)
    N = x.node_padding_mask.shape[0]

    assert output[0].shape == (N, hidden_size)
    assert output[1].shape == (N, max_nei, hidden_size)
