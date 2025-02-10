from functools import partial

import torch
import torch_geometric

from fairchem.core.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

from ..custom_types import GraphAttentionData, GeneralGraphAttentionData
from ..configs import (
    GlobalConfigs,
    MolecularGraphConfigs,
    GeneralMolecularGraphConfigs,
    GraphNeuralNetworksConfigs,
)
from .graph_utils import (
    get_node_direction_expansion,
    convert_neighbor_list,
    map_neighbor_list,
    get_attn_mask,
    patch_singleton_atom,
    pad_batch,
    pad_batch_pos,
    one_hot_encode,
)


def data_preprocess(
    data,
    generate_graph_fn: callable,
    global_cfg: GlobalConfigs,
    gnn_cfg: GraphNeuralNetworksConfigs,
    molecular_graph_cfg: MolecularGraphConfigs,
) -> GraphAttentionData:
    # atomic numbers
    atomic_numbers = data.atomic_numbers.long()

    # edge distance expansion
    expansion_func = {
        "gaussian": GaussianSmearing,
        "sigmoid": SigmoidSmearing,
        "linear_sigmoid": LinearSigmoidSmearing,
        "silu": SiLUSmearing,
    }[molecular_graph_cfg.distance_function]

    edge_distance_expansion_func = expansion_func(
        0.0,
        molecular_graph_cfg.max_radius,
        gnn_cfg.edge_distance_expansion_size,
        basis_width_scalar=2.0,
    ).to(data.pos.device)

    # generate graph
    graph = generate_graph_fn(data)

    # sort edge index according to receiver node
    edge_index, edge_attr = torch_geometric.utils.sort_edge_index(
        graph.edge_index,
        [graph.edge_distance, graph.edge_distance_vec],
        sort_by_row=False,
    )
    edge_distance, edge_distance_vec = edge_attr[0], edge_attr[1]

    # edge directions (for direct force prediction, ref: gemnet)
    edge_direction = -edge_distance_vec / edge_distance[:, None]

    # edge distance expansion (ref: scn)
    edge_distance_expansion = edge_distance_expansion_func(edge_distance)

    # node direction expansion
    node_direction_expansion = get_node_direction_expansion(
        distance_vec=edge_distance_vec,
        edge_index=edge_index,
        lmax=gnn_cfg.node_direction_expansion_size - 1,
        num_nodes=data.num_nodes,
    )

    # convert to neighbor list
    neighbor_list, neighbor_mask, index_mapping = convert_neighbor_list(
        edge_index, molecular_graph_cfg.max_neighbors, data.num_nodes
    )

    # map neighbor list
    map_neighbor_list_ = partial(
        map_neighbor_list,
        index_mapping=index_mapping,
        max_neighbors=molecular_graph_cfg.max_neighbors,
        num_nodes=data.num_nodes,
    )
    edge_direction = map_neighbor_list_(edge_direction)
    edge_distance_expansion = map_neighbor_list_(edge_distance_expansion)

    # pad batch
    if global_cfg.use_padding:
        (
            atomic_numbers,
            node_direction_expansion,
            edge_distance_expansion,
            edge_direction,
            neighbor_list,
            neighbor_mask,
            node_batch,
            node_padding_mask,
            graph_padding_mask,
        ) = pad_batch(
            max_num_nodes_per_batch=molecular_graph_cfg.max_num_nodes_per_batch,
            atomic_numbers=atomic_numbers,
            node_direction_expansion=node_direction_expansion,
            edge_distance_expansion=edge_distance_expansion,
            edge_direction=edge_direction,
            neighbor_list=neighbor_list,
            neighbor_mask=neighbor_mask,
            node_batch=data.batch,
            num_graphs=data.num_graphs,
            batch_size=global_cfg.batch_size,
        )
    else:
        node_padding_mask = torch.ones_like(atomic_numbers, dtype=torch.bool)
        graph_padding_mask = torch.ones(
            data.num_graphs, dtype=torch.bool, device=data.batch.device
        )
        node_batch = data.batch

    # patch singleton atom
    edge_direction, neighbor_list, neighbor_mask = patch_singleton_atom(
        edge_direction, neighbor_list, neighbor_mask
    )

    # get attention mask
    attn_mask, angle_embedding = get_attn_mask(
        edge_direction=edge_direction,
        neighbor_mask=neighbor_mask,
        num_heads=gnn_cfg.atten_num_heads,
        use_angle_embedding=gnn_cfg.use_angle_embedding,
    )

    if gnn_cfg.atten_name in ["memory_efficient", "flash", "math"]:
        torch.backends.cuda.enable_flash_sdp(gnn_cfg.atten_name == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(
            gnn_cfg.atten_name == "memory_efficient"
        )
        torch.backends.cuda.enable_math_sdp(gnn_cfg.atten_name == "math")
    else:
        raise NotImplementedError(
            f"Attention name {gnn_cfg.atten_name} not implemented"
        )

    # construct input data
    x = GraphAttentionData(
        atomic_numbers=atomic_numbers,
        node_direction_expansion=node_direction_expansion,
        edge_distance_expansion=edge_distance_expansion,
        edge_direction=edge_direction,
        attn_mask=attn_mask,
        angle_embedding=angle_embedding,
        neighbor_list=neighbor_list,
        neighbor_mask=neighbor_mask,
        node_batch=node_batch,
        node_padding_mask=node_padding_mask,
        graph_padding_mask=graph_padding_mask,
    )
    return x


def data_preprocess_spin_charge(
    data,
    generate_graph_fn: callable,
    generate_graph_fn_lr: callable,
    global_cfg: GlobalConfigs,
    gnn_cfg: GraphNeuralNetworksConfigs,
    molecular_graph_cfg: GeneralMolecularGraphConfigs,
) -> GeneralGraphAttentionData:
    # atomic numbers
    atomic_numbers = data.atomic_numbers.long()

    # positions
    pos = data.pos

    # edge distance expansion
    expansion_func = {
        "gaussian": GaussianSmearing,
        "sigmoid": SigmoidSmearing,
        "linear_sigmoid": LinearSigmoidSmearing,
        "silu": SiLUSmearing,
    }[molecular_graph_cfg.distance_function]

    edge_distance_expansion_func = expansion_func(
        0.0,
        molecular_graph_cfg.max_radius,
        gnn_cfg.edge_distance_expansion_size,
        basis_width_scalar=2.0,
    ).to(data.pos.device)

    # generate graph
    graph = generate_graph_fn(data)
    graph_lr = generate_graph_fn_lr(data)
    edge_index_lr = graph_lr.edge_index

    # if molecular_graph_cfg.max_radius == molecular_graph_cfg.max_radius_lr:
    #    #TODO: reuse radius graph potentially? this is performance tho
    #    pass

    # sort edge index according to receiver node
    edge_index, edge_attr = torch_geometric.utils.sort_edge_index(
        graph.edge_index,
        [graph.edge_distance, graph.edge_distance_vec],
        sort_by_row=False,
    )
    edge_distance, edge_distance_vec = edge_attr[0], edge_attr[1]

    # edge directions (for direct force prediction, ref: gemnet)
    edge_direction = -edge_distance_vec / edge_distance[:, None]

    # edge distance expansion (ref: scn)
    edge_distance_expansion = edge_distance_expansion_func(edge_distance)

    # node direction expansion
    node_direction_expansion = get_node_direction_expansion(
        distance_vec=edge_distance_vec,
        edge_index=edge_index,
        lmax=gnn_cfg.node_direction_expansion_size - 1,
        num_nodes=data.num_nodes,
    )

    # convert to neighbor list
    neighbor_list, neighbor_mask, index_mapping = convert_neighbor_list(
        edge_index, molecular_graph_cfg.max_neighbors, data.num_nodes
    )

    # map neighbor list
    map_neighbor_list_ = partial(
        map_neighbor_list,
        index_mapping=index_mapping,
        max_neighbors=molecular_graph_cfg.max_neighbors,
        num_nodes=data.num_nodes,
    )
    edge_direction = map_neighbor_list_(edge_direction)
    edge_distance_expansion = map_neighbor_list_(edge_distance_expansion)

    n_nodes_graphs = data.ptr.diff()

    if molecular_graph_cfg.allowed_spins:
        # spin might be a non-consecutive integers, sort
        # spin_sorted = torch.sort(data.spin).values

        # create spin tensor
        spin = data.spin
        spin_expand = one_hot_encode(
            spin,
            n_nodes_graphs,
            torch.tensor(molecular_graph_cfg.allowed_spins, device=data.batch.device),
        )
        # concat on to node direction expansion
        node_direction_expansion = torch.cat(
            (node_direction_expansion, spin_expand), dim=1
        )

    if molecular_graph_cfg.allowed_charges:
        # create charge tensor
        charge = data.charge
        charge_expand = one_hot_encode(
            charge,
            n_nodes_graphs,
            torch.tensor(molecular_graph_cfg.allowed_charges, device=data.batch.device),
        )
        # concat on to node direction expansion
        node_direction_expansion = torch.cat(
            (node_direction_expansion, charge_expand), dim=1
        )

    # pad batch
    if global_cfg.use_padding:
        (
            atomic_numbers,
            node_direction_expansion,
            edge_distance_expansion,
            edge_direction,
            neighbor_list,
            neighbor_mask,
            node_batch,
            node_padding_mask,
            graph_padding_mask,
            _,
        ) = pad_batch_pos(
            max_num_nodes_per_batch=molecular_graph_cfg.max_num_nodes_per_batch,
            atomic_numbers=atomic_numbers,
            node_direction_expansion=node_direction_expansion,
            edge_distance_expansion=edge_distance_expansion,
            edge_direction=edge_direction,
            neighbor_list=neighbor_list,
            neighbor_mask=neighbor_mask,
            node_batch=data.batch,
            num_graphs=data.num_graphs,
            batch_size=global_cfg.batch_size,
            pos=None,
        )
    else:
        node_padding_mask = torch.ones_like(atomic_numbers, dtype=torch.bool)
        graph_padding_mask = torch.ones(
            data.num_graphs, dtype=torch.bool, device=data.batch.device
        )
        node_batch = data.batch

    # patch singleton atom
    edge_direction, neighbor_list, neighbor_mask = patch_singleton_atom(
        edge_direction, neighbor_list, neighbor_mask
    )

    # get attention mask
    attn_mask, angle_embedding = get_attn_mask(
        edge_direction=edge_direction,
        neighbor_mask=neighbor_mask,
        num_heads=gnn_cfg.atten_num_heads,
        use_angle_embedding=gnn_cfg.use_angle_embedding,
    )

    if gnn_cfg.atten_name in ["memory_efficient", "flash", "math"]:
        torch.backends.cuda.enable_flash_sdp(gnn_cfg.atten_name == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(
            gnn_cfg.atten_name == "memory_efficient"
        )
        torch.backends.cuda.enable_math_sdp(gnn_cfg.atten_name == "math")
    else:
        raise NotImplementedError(
            f"Attention name {gnn_cfg.atten_name} not implemented"
        )

    x = GeneralGraphAttentionData(
        atomic_numbers=atomic_numbers,
        node_direction_expansion=node_direction_expansion,
        edge_distance_expansion=edge_distance_expansion,
        edge_direction=edge_direction,
        attn_mask=attn_mask,
        angle_embedding=angle_embedding,
        neighbor_list=neighbor_list,
        neighbor_mask=neighbor_mask,
        node_batch=node_batch,
        node_padding_mask=node_padding_mask,
        graph_padding_mask=graph_padding_mask,
        spin=data.spin,
        charge=data.charge,
        partial_charge=None,  # TODO: use this if we ever get here
        partial_spin=None,  # TODO: use this if we ever get here
        pos=pos,
        edge_index_lr=edge_index_lr,
    )
    return x
