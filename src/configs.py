from dataclasses import dataclass, fields, is_dataclass, field
from typing import Any, Dict, Literal, Type, Optional


@dataclass
class GlobalConfigs:
    regress_forces: bool
    direct_force: bool
    hidden_size: int  # divisible by 2 and num_heads
    batch_size: int
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ]
    use_compile: bool = True
    use_padding: bool = True
    j_coupling_hidden_dim: int = 128
    use_dipole: bool = False
    dipole_key: Optional[str] = "dipole_moment"


@dataclass
class MolecularGraphConfigs:
    use_pbc: bool
    use_pbc_single: bool
    otf_graph: bool
    max_neighbors: int
    max_radius: float
    max_num_elements: int
    max_num_nodes_per_batch: int
    enforce_max_neighbors_strictly: bool
    distance_function: Literal["gaussian", "sigmoid", "linearsigmoid", "silu"]
    max_radius_lr: float = 8.0
    max_neighbors_lr: int = 30
    allowed_charges: list = field(default_factory=list)
    allowed_spins: list = field(default_factory=list)
    use_partial_charge: bool = False
    use_partial_spin: bool = False


@dataclass
class GraphNeuralNetworksConfigs:
    num_layers: int
    atom_embedding_size: int
    node_direction_embedding_size: int
    node_direction_expansion_size: int
    edge_distance_expansion_size: int
    edge_distance_embedding_size: int
    atten_name: Literal[
        "math",
        "memory_efficient",
        "flash",
    ]
    atten_num_heads: int
    readout_hidden_layer_multiplier: int
    output_hidden_layer_multiplier: int
    ffn_hidden_layer_multiplier: int
    use_angle_embedding: bool = True
    energy_reduce: Literal["sum", "mean"] = "sum"
    constrain_charge: bool = False
    constrain_spin: bool = False
    two_component_latent_charge: bool = False
    heisenberg_tf: bool = False


@dataclass
class RegularizationConfigs:
    mlp_dropout: float
    atten_dropout: float
    stochastic_depth_prob: float
    normalization: Literal["layernorm", "rmsnorm", "skip"]


@dataclass
class EScAIPConfigs:
    global_cfg: GlobalConfigs
    molecular_graph_cfg: MolecularGraphConfigs
    gnn_cfg: GraphNeuralNetworksConfigs
    reg_cfg: RegularizationConfigs


def init_configs(cls: Type[EScAIPConfigs], kwargs: Dict[str, Any]) -> EScAIPConfigs:
    """
    Initialize a dataclass with the given kwargs.
    """
    init_kwargs = {}
    for field_temp in fields(cls):
        # print(field_temp.name, field_temp.type)
        if is_dataclass(field_temp.type):
            init_kwargs[field_temp.name] = init_configs(field_temp.type, kwargs)
        elif field_temp.name in kwargs:
            init_kwargs[field_temp.name] = kwargs[field_temp.name]
        elif field_temp.default is not None:
            init_kwargs[field_temp.name] = field_temp.default
        else:
            raise ValueError(
                f"Missing required configuration parameter: '{field_temp.name}'"
            )

    return cls(**init_kwargs)
