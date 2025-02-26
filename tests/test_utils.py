import torch
import yaml
import lmdb

import pickle as pkl
from fairchem.core.models.base import HydraModel
from torch_geometric.data import Batch
from functools import partial

from src.utils.graph_utils import one_hot_encode
from src.configs import MolecularGraphConfigs
from src.utils.data_preprocess import data_preprocess_spin_charge


def load_data_model(
    data_path: str = "tests/data/OC20_batch_3.pt",
    model_path: str = "tests/data/L2_H4_64_.yml",
):
    batch = torch.load(data_path)
    with open(model_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    # print("config: ", config['model'])
    model = HydraModel(**config["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return batch.to(device), model.to(device)


def load_data_model_general(
    model_path: str = "tests/data/L2_H4_64_general.yml",
):
    batch = load_lmdb()
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
    new_mol_config = MolecularGraphConfigs(
        use_pbc=molecular_graph_cfg.use_pbc,
        use_pbc_single=molecular_graph_cfg.use_pbc_single,
        otf_graph=molecular_graph_cfg.otf_graph,
        max_neighbors=molecular_graph_cfg.max_neighbors,
        max_radius=molecular_graph_cfg.max_radius,
        max_radius_lr=molecular_graph_cfg.max_radius_lr,
        max_num_elements=molecular_graph_cfg.max_num_elements,
        max_num_nodes_per_batch=molecular_graph_cfg.max_num_nodes_per_batch,
        enforce_max_neighbors_strictly=molecular_graph_cfg.enforce_max_neighbors_strictly,
        allowed_charges=[0, 1, 2],
        allowed_spins=[0, 1, 2],
        use_partial_charge=False,
        use_partial_spin=False,
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


def load_general_model_configs():
    dict_model = {
        "trainer": "equiformerv2_forces",
        "dataset": {
            "train": {
                "format": "lmdb",
                "src": "/home/santiagovargas/dev/EScAIP/dev/data/MD22_lmdb/DHA/train/",
                "key_mapping": {"y": "energy", "force": "forces"},
                "transforms": {
                    "normalizer": {
                        "energy": {"mean": -27383.035, "stdev": 0.41342595},
                        "forces": {"mean": 0, "stdev": 1.1258113},
                    }
                },
            },
            "val": {
                "src": "/home/santiagovargas/dev/EScAIP/dev/data/MD22_lmdb/DHA/val/"
            },
        },
        "outputs": {
            "energy": {"shape": 1, "level": "system"},
            "forces": {
                "irrep_dim": 1,
                "level": "atom",
                "train_on_free_atoms": True,
                "eval_on_free_atoms": True,
            },
        },
        "loss_functions": [
            {"energy": {"fn": "mae", "coefficient": 1}},
            {"forces": {"fn": "l2mae", "coefficient": 100}},
        ],
        "evaluation_metrics": {
            "metrics": {
                "energy": ["mae"],
                "forces": ["mae", "cosine_similarity", "magnitude_error"],
                "misc": ["energy_forces_within_threshold"],
            },
            "primary_metric": "forces_mae",
        },
        "model": {
            "name": "hydra",
            "pass_through_head_outputs": True,
            "otf_graph": True,
            "backbone": {
                "model": "src.EScAIP.GeneralEScAIPBackbone",
                "activation": "gelu",
                "direct_force": True,
                "hidden_size": 512,
                "regress_forces": True,
                "use_fp16_backbone": False,
                "batch_size": 32,
                "enforce_max_neighbors_strictly": True,
                "distance_function": "gaussian",
                "max_neighbors": 20,
                "max_num_elements": 90,
                "max_num_nodes_per_batch": 64,
                "max_radius": 6.0,
                "otf_graph": True,
                "use_pbc": True,
                "use_pbc_single": False,
                "atom_embedding_size": 128,
                "atten_name": "memory_efficient",
                "atten_num_heads": 8,
                "edge_distance_embedding_size": 512,
                "edge_distance_expansion_size": 600,
                "node_direction_embedding_size": 64,
                "node_direction_expansion_size": 10,
                "num_layers": 6,
                "output_hidden_layer_multiplier": 2,
                "readout_hidden_layer_multiplier": 2,
                "ffn_hidden_layer_multiplier": 2,
                "use_angle_embedding": True,
                "atten_dropout": 0.1,
                "mlp_dropout": 0.05,
                "normalization": "rmsnorm",
                "stochastic_depth_prob": 0.0,
                "allowed_spins": [1, 2, 3],
                "allowed_charges": [-2, -1, 0, 1, 2],
                "use_partial_charge": False,
                "use_partial_spin": False,
            },
            "heads": {
                "forces": {"module": "src.EScAIP.EScAIPDirectForceHead"},
                "energy": {"module": "src.EScAIP.EScAIPEnergyHead"},
            },
        },
        "optim": {
            "batch_size": 32,
            "eval_batch_size": 24,
            "num_workers": 8,
            "lr_initial": 4e-05,
            "optimizer": "AdamW",
            "optimizer_params": {"weight_decay": 0.01},
            "scheduler": "LambdaLR",
            "scheduler_params": {
                "lambda_type": "cosine",
                "warmup_factor": 0.2,
                "warmup_epochs": 0.1,
                "lr_min_factor": 0.5,
            },
            "max_epochs": 100,
            "clip_grad_norm": 10,
            "ema_decay": 0.999,
            "eval_every": 100,
        },
        "mode": "train",
        "identifier": "",
        "timestamp_id": None,
        "seed": 0,
        "is_debug": False,
        "run_dir": "./dev/training/v0",
        "print_every": 10,
        "amp": False,
        "checkpoint": None,
        "cpu": False,
        "submit": False,
        "world_size": 1,
        "distributed_backend": "nccl",
        "gp_gpus": None,
    }
    return dict_model
