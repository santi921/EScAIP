import torch
import yaml
import lmdb

import pickle as pkl
from fairchem.core.models.base import HydraModel
from torch_geometric.data import Batch
from functools import partial

from src.utils.graph_utils import one_hot_encode
from src.configs import GeneralMolecularGraphConfigs
from src.utils.data_preprocess import data_preprocess_spin_charge


def load_data_model(
    data_path: str = "tests/data/OC20_batch_3.pt",
    model_path: str = "tests/data/L2_H4_64_.yml",
):
    batch = torch.load(data_path)
    with open(model_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    model = HydraModel(**config["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return batch.to(device), model.to(device)


def load_one_hot():
    n_nodes = torch.tensor([15, 17, 17])
    global_data = torch.tensor([-1, 0, 2])
    allowed_values = torch.tensor([-1, 0, 1, 2, 4])
    one_hot_data = one_hot_encode(global_data, n_nodes, allowed_values)
    return one_hot_data


def load_lmdb():
    db = lmdb.open(
        "tests/data/lmdb/data/data.lmdb",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    list_data = []
    # iterate through db
    ind = 0
    with db.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            val_decode = pkl.loads(value)
            if str(key)[2:-1] != "length":
                list_data.append(val_decode)
            # print(val_decode.spin, val_decode.charge, val_decode.num_nodes)
            # add b/c model has low batch size
            if ind == 2:
                break
            ind += 1
        batched_data = Batch.from_data_list(list_data)

    return batched_data


def load_model():
    model_path: str = "tests/data/L2_H4_64_.yml"
    with open(model_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model = HydraModel(**config["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def load_preprocess_example():
    batched_data = load_lmdb()
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batched_data.to(device)

    global_cfg = model.backbone.global_cfg
    gnn_cfg = model.backbone.gnn_cfg
    molecular_graph_cfg = model.backbone.molecular_graph_cfg

    generate_graph_fn = partial(
        model.backbone.generate_graph,
        cutoff=molecular_graph_cfg.max_radius,
        max_neighbors=molecular_graph_cfg.max_neighbors,
        use_pbc=molecular_graph_cfg.use_pbc,
        otf_graph=molecular_graph_cfg.otf_graph,
        enforce_max_neighbors_strictly=molecular_graph_cfg.enforce_max_neighbors_strictly,
        use_pbc_single=molecular_graph_cfg.use_pbc_single,
    )

    # TODO: replace once the new global config is implemented
    new_mol_config = GeneralMolecularGraphConfigs(
        use_pbc=molecular_graph_cfg.use_pbc,
        use_pbc_single=molecular_graph_cfg.use_pbc_single,
        otf_graph=molecular_graph_cfg.otf_graph,
        max_neighbors=molecular_graph_cfg.max_neighbors,
        max_radius=molecular_graph_cfg.max_radius,
        max_num_elements=molecular_graph_cfg.max_num_elements,
        max_num_nodes_per_batch=molecular_graph_cfg.max_num_nodes_per_batch,
        enforce_max_neighbors_strictly=molecular_graph_cfg.enforce_max_neighbors_strictly,
        allowed_charges=[0, 1, 2],
        allowed_spins=[0, 1, 2],
        distance_function=molecular_graph_cfg.distance_function,
    )

    processed = data_preprocess_spin_charge(
        data=batched_data,
        generate_graph_fn=generate_graph_fn,
        global_cfg=global_cfg,
        gnn_cfg=gnn_cfg,
        molecular_graph_cfg=new_mol_config,
    )

    return processed
