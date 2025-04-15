import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from e3nn.o3._spherical_harmonics import _spherical_harmonics


@torch.jit.script
def get_node_direction_expansion(
    distance_vec: torch.Tensor, edge_index: torch.Tensor, lmax: int, num_nodes: int
):
    """
    Calculate Bond-Orientational Order (BOO) for each node in the graph.
    Ref: Steinhardt, et al. "Bond-orientational order in liquids and glasses." Physical Review B 28.2 (1983): 784.
    Return: (N, )
    """
    distance_vec = torch.nn.functional.normalize(distance_vec, dim=-1)
    edge_sh = _spherical_harmonics(
        lmax=lmax,
        x=distance_vec[:, 0],
        y=distance_vec[:, 1],
        z=distance_vec[:, 2],
    )
    node_boo = torch.zeros((num_nodes, edge_sh.shape[1]), device=edge_sh.device)
    node_boo = scatter(edge_sh, edge_index[1], dim=0, out=node_boo, reduce="mean")
    sh_index = torch.arange(lmax + 1, device=node_boo.device)
    sh_index = torch.repeat_interleave(sh_index, 2 * sh_index + 1)
    node_boo = scatter(node_boo**2, sh_index, dim=1, reduce="sum").sqrt()
    return node_boo


def convert_neighbor_list(edge_index: torch.Tensor, max_neighbors: int, num_nodes: int):
    """
    Convert edge_index to a neighbor list format.
    """
    src = edge_index[0, :]
    dst = edge_index[1, :]

    # Count the number of neighbors for each node
    # print("dst: ",dst)
    neighbor_counts = torch.bincount(dst, minlength=num_nodes)
    # print("neighbor counts: ", neighbor_counts)
    # Calculate the offset for each node
    offset = max_neighbors - neighbor_counts
    # print("offset: ", offset)
    # print("offset min: ", offset.min(), "offset max: ", offset.max())
    # print(offset)
    offset = torch.cat(
        [torch.tensor([0], device=offset.device), torch.cumsum(offset, dim=0)]
    )

    # Create an index mapping
    index_mapping = torch.arange(0, edge_index.shape[1], device=edge_index.device)

    # Calculate the indices in the neighbor list
    index_mapping = offset[dst] + index_mapping

    # Initialize the neighbor list and mask
    neighbor_list = torch.full(
        (num_nodes * max_neighbors,), -1, dtype=torch.long, device=edge_index.device
    )
    mask = torch.zeros(
        (num_nodes * max_neighbors,), dtype=torch.bool, device=edge_index.device
    )

    # Scatter the neighbors
    # print("index mapping: ", index_mapping)
    # print("src: ", src)
    neighbor_list.scatter_(0, index_mapping, src)
    mask.scatter_(
        0,
        index_mapping,
        torch.ones_like(src, dtype=torch.bool, device=edge_index.device),
    )

    # Reshape to [N, max_num_neighbors]
    neighbor_list = neighbor_list.view(num_nodes, max_neighbors)
    mask = mask.view(num_nodes, max_neighbors)

    return neighbor_list, mask, index_mapping


def map_neighbor_list(x, index_mapping, max_neighbors, num_nodes):
    """
    Map from edges to neighbor list.
    x: (num_edges, h)
    index_mapping: (num_edges, )
    return: (num_nodes, max_neighbors, h)
    """
    output = torch.zeros((num_nodes * max_neighbors, x.shape[1]), device=x.device)
    output.scatter_(0, index_mapping.unsqueeze(1).expand(-1, x.shape[1]), x)
    return output.view(num_nodes, max_neighbors, x.shape[1])


def map_sender_receiver_feature(sender_feature, receiver_feature, neighbor_list):
    """
    Map from node features to edge features.
    sender_feature, receiver_feature: (num_nodes, h)
    neighbor_list: (num_nodes, max_neighbors)
    return: sender_features, receiver_features (num_nodes, max_neighbors, h)
    """
    # sender feature
    sender_feature = sender_feature[neighbor_list.flatten()].view(
        neighbor_list.shape[0], neighbor_list.shape[1], -1
    )

    # receiver features
    receiver_feature = receiver_feature.unsqueeze(1).expand(
        -1, neighbor_list.shape[1], -1
    )

    return (sender_feature, receiver_feature)


# @torch.compile
def get_attn_mask(
    edge_direction: torch.Tensor,
    neighbor_mask: torch.Tensor,
    num_heads: int,
    use_angle_embedding: bool,
):
    # create a mask for empty neighbors
    batch_size, max_neighbors = neighbor_mask.shape
    attn_mask = torch.zeros(
        batch_size, max_neighbors, max_neighbors, device=neighbor_mask.device
    )
    attn_mask = attn_mask.masked_fill(~neighbor_mask.unsqueeze(1), float("-inf"))

    # repeat the mask for each head
    attn_mask = (
        attn_mask.unsqueeze(1)
        .expand(batch_size, num_heads, max_neighbors, max_neighbors)
        .reshape(batch_size * num_heads, max_neighbors, max_neighbors)
    )

    # get the angle embeddings
    dot_product = torch.matmul(edge_direction, edge_direction.transpose(1, 2))
    dot_product = (
        dot_product.unsqueeze(1)
        .expand(-1, num_heads, -1, -1)
        .reshape(batch_size * num_heads, max_neighbors, max_neighbors)
    )

    return attn_mask, dot_product


def pad_batch(
    max_num_nodes_per_batch,
    atomic_numbers,
    node_direction_expansion,
    edge_distance_expansion,
    edge_direction,
    neighbor_list,
    neighbor_mask,
    node_batch,
    num_graphs,
    batch_size,
):
    """
    Pad the batch to have the same number of nodes in total.
    Needed for torch.compile

    Note: the sampler for multi-node training could sample batchs with different number of graphs.
    The number of sampled graphs could be smaller or larger than the batch size.
    This would cause the model to recompile or core dump.
    Temporarily, setting the max number of graphs to be twice the batch size to mitigate this issue.
    TODO: look into a better way to handle this.
    """
    device = atomic_numbers.device
    num_nodes, _ = neighbor_list.shape
    pad_size = max_num_nodes_per_batch * batch_size - num_nodes
    assert (
        pad_size >= 0
    ), "Number of nodes exceeds the maximum number of nodes per batch"

    # pad the features
    atomic_numbers = F.pad(atomic_numbers, (0, pad_size), value=0)
    node_direction_expansion = F.pad(
        node_direction_expansion, (0, 0, 0, pad_size), value=0
    )
    edge_distance_expansion = F.pad(
        edge_distance_expansion, (0, 0, 0, 0, 0, pad_size), value=0
    )
    edge_direction = F.pad(edge_direction, (0, 0, 0, 0, 0, pad_size), value=0)
    neighbor_list = F.pad(neighbor_list, (0, 0, 0, pad_size), value=-1)
    neighbor_mask = F.pad(neighbor_mask, (0, 0, 0, pad_size), value=0)
    node_batch = F.pad(node_batch, (0, pad_size), value=num_graphs)

    # create the padding mask
    node_padding_mask = torch.ones(
        max_num_nodes_per_batch * batch_size, dtype=torch.bool, device=device
    )
    node_padding_mask[num_nodes:] = False

    # TODO look into a better way to handle this
    # is this 2x just random?
    graph_padding_mask = torch.ones(batch_size * 2, dtype=torch.bool, device=device)
    graph_padding_mask[num_graphs:] = False

    return (
        atomic_numbers,
        node_direction_expansion,
        edge_distance_expansion,
        edge_direction,
        neighbor_list,
        neighbor_mask,
        node_batch,
        node_padding_mask,
        graph_padding_mask,
    )


def pad_batch_pos(
    max_num_nodes_per_batch,
    atomic_numbers,
    node_direction_expansion,
    edge_distance_expansion,
    edge_direction,
    neighbor_list,
    neighbor_mask,
    node_batch,
    num_graphs,
    batch_size,
    pos=None,
):
    """
    Pad the batch to have the same number of nodes in total.
    Needed for torch.compile

    Note: the sampler for multi-node training could sample batchs with different number of graphs.
    The number of sampled graphs could be smaller or larger than the batch size.
    This would cause the model to recompile or core dump.
    Temporarily, setting the max number of graphs to be twice the batch size to mitigate this issue.
    TODO: look into a better way to handle this.
    """
    device = atomic_numbers.device
    num_nodes, _ = neighbor_list.shape
    pad_size = max_num_nodes_per_batch * batch_size - num_nodes
    assert (
        pad_size >= 0
    ), "Number of nodes exceeds the maximum number of nodes per batch"

    # pad the features
    atomic_numbers = F.pad(atomic_numbers, (0, pad_size), value=0)
    node_direction_expansion = F.pad(
        node_direction_expansion, (0, 0, 0, pad_size), value=0
    )
    edge_distance_expansion = F.pad(
        edge_distance_expansion, (0, 0, 0, 0, 0, pad_size), value=0
    )
    edge_direction = F.pad(edge_direction, (0, 0, 0, 0, 0, pad_size), value=0)
    neighbor_list = F.pad(neighbor_list, (0, 0, 0, pad_size), value=-1)
    neighbor_mask = F.pad(neighbor_mask, (0, 0, 0, pad_size), value=0)
    node_batch = F.pad(node_batch, (0, pad_size), value=num_graphs)

    # knock this out for now
    # if pos is not None:
    #    pos = F.pad(pos, (0, 0, 0, pad_size), value=0)

    # create the padding mask
    node_padding_mask = torch.ones(
        max_num_nodes_per_batch * batch_size, dtype=torch.bool, device=device
    )
    node_padding_mask[num_nodes:] = False

    # TODO look into a better way to handle this
    # is this 2x just random?
    graph_padding_mask = torch.ones(batch_size * 2, dtype=torch.bool, device=device)
    graph_padding_mask[num_graphs:] = False

    return (
        atomic_numbers,
        node_direction_expansion,
        edge_distance_expansion,
        edge_direction,
        neighbor_list,
        neighbor_mask,
        node_batch,
        node_padding_mask,
        graph_padding_mask,
        pos,
    )


def unpad_results(results, node_padding_mask, graph_padding_mask):
    """
    Unpad the results to remove the padding.
    """
    unpad_results = {}
    for key in results:
        if results[key].shape[0] == node_padding_mask.shape[0]:
            unpad_results[key] = results[key][node_padding_mask]
        elif results[key].shape[0] == graph_padding_mask.shape[0]:
            unpad_results[key] = results[key][graph_padding_mask]
        else:
            raise ValueError("Unknown padding mask shape")
    return unpad_results


def patch_singleton_atom(edge_direction, neighbor_list, neighbor_mask):
    """
    Patch the singleton atoms in the neighbor_list and neighbor_mask.
    Add a self-loop to the singleton atom
    """

    # Find the singleton atoms
    idx = torch.where(neighbor_mask.sum(dim=-1) == 0)[0]

    # patch edge_direction to unit vector
    edge_direction[idx, 0] = torch.tensor(
        [1.0, 0.0, 0.0], device=edge_direction.device, dtype=edge_direction.dtype
    )

    # patch neighbor_list to itself
    neighbor_list[idx, 0] = idx

    # patch neighbor_mask to itself
    neighbor_mask[idx, 0] = 1

    return edge_direction, neighbor_list, neighbor_mask


def compilable_scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    dim: int = 0,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    torch_scatter scatter function with compile support.
    Modified from torch_geometric.utils.scatter_.
    """
    reduce = reduce.lower()
    # print("reduce", reduce)

    def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
        dim = ref.dim() + dim if dim < 0 else dim
        size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
        return src.view(size).expand_as(ref)

    # print("dim_size", dim_size)
    dim = src.dim() + dim if dim < 0 else dim
    size = src.size()[:dim] + (dim_size,) + src.size()[dim + 1 :]
    # print("reduce", reduce)
    if reduce == "sum" or reduce == "add":
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    if reduce == "mean":
        # print("mean")
        count = src.new_zeros(dim_size)
        # print("new zeros") breaks here
        count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        # print("scatter add")
        count = count.clamp(min=1)
        # print("pre broadcast")
        index = broadcast(index, src, dim)
        # this errors out sometimes

        out = src.new_zeros(size).scatter_add_(dim, index, src)
        # print("final out")
        return out / broadcast(count, out, dim)

    if reduce == "None":
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_(dim, index, src)

    raise ValueError((f"Invalid reduce option '{reduce}'."))


@torch.jit.script
def one_hot_encode(
    data: torch.Tensor, n_nodes: torch.Tensor, allowed_values: torch.Tensor
) -> torch.Tensor:
    """
    One-hot encode the data.
    Takes:
        data: (num_graphs, )
        n_nodes: (num_graphs, )
        allowed_values: (num_values, )
    Returns:
        data: (sum(n_nodes), num_values)
    """
    list_data = []
    ind = 0
    for i in n_nodes:
        global_temp = data[ind]
        data_temp = global_temp.expand(i).reshape(-1)
        data_temp = (data_temp.unsqueeze(1) == allowed_values).float()
        list_data.append(data_temp)
        ind += 1
    data = torch.cat(list_data)
    return data


# @torch.compiler.disable()
# @torch.jit.script
def get_potential(
    q: torch.Tensor,
    convergence_func: torch.Tensor,
    edge_dist_transformed: torch.Tensor,
    ind_charges: torch.Tensor,
    ind_interactions: torch.Tensor,
):
    """
    Get the potential energy for each atom. Compilable function.
    Takes:
    q: charge vector of shape (n_atoms, 1)
    convergence_func: convergence function of shape (n_atoms, n_neighbors)
    edge_dist_transformed: transformed distance matrix of shape (n_atoms, n_neighbors)
    ind_charges: list of charges for each atom of shape (n_atoms, 1)
    ind_interactions: list of interactions for each atom of shape (n_atoms, n_neighbors)
    """
    q_now = q[ind_charges]
    convergence_func_now = convergence_func[ind_interactions]
    edge_dist_transformed_now = edge_dist_transformed[ind_interactions]
    pairwise = q_now * q_now * edge_dist_transformed_now * convergence_func_now
    # alternatively compute with einsum
    # q_now_2 = torch.pow(q_now, 2).view(-1)
    # print(q_now_2.shape, edge_dist_transformed_now.shape, convergence_func_now.shape)
    # pairwise = torch.einsum('i,i,i->i', q_now_2, edge_dist_transformed_now, #convergence_func_now)
    return pairwise.sum().view(-1)


# @torch.jit.script
def potential_full_from_edge_inds(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    q: torch.Tensor,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    twopi: float = 2.0 * np.pi,
):
    """
    Get the potential energy for each atom in the batch.
    Takes:
        pos: position matrix of shape (n_atoms, 3)
        edge_index: edge index of shape (2, n_edges)
        q: charge vector of shape (n_atoms, 1)
        radius_lr: cutoff radius for long-range interactions
        sigma: sigma parameter for the error function
        epsilon: epsilon parameter for the error function
        twopi: 2 * pi
        max_num_neighbors: maximum number of neighbors for each atom
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """
    # batch uses pos, batch. That's it

    # yields list of interactions [source, target]
    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    # red to [n_interactions, 1]
    edge_dist = distance_vec.norm(dim=-1)

    edge_dist_transformed = (1.0 / (edge_dist + epsilon)) / twopi / 2.0
    convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5))
    # print(convergence_func)
    # q = q.view(-1)

    q_source = q[i].view(-1)
    q_target = q[j].view(-1)
    pairwise_potential = q_source * q_target * edge_dist_transformed * convergence_func

    results = scatter(pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum")

    return results


def heisenberg_potential_full_from_edge_inds(
    pos: torch.Tensor,  # keep
    edge_index: torch.Tensor,  # keep
    q: torch.Tensor,
    nn: torch.nn.Module,
    sigma: float = 1.0,
):
    """
    Get the potential energy for each atom in the batch.
    Takes:
        pos: position matrix of shape (n_atoms, 3)
        edge_index: edge index of shape (2, n_edges)
        q: charge vector of shape (n_atoms, 2)
        nn: neural network to calculate the coupling term
        sigma: sigma parameter for the error function
        epsilon: epsilon parameter for the error function
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """
    # batch uses pos, batch. That's it
    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    edge_dist = distance_vec.norm(dim=-1).reshape(-1, 1)
    edge_dist.requires_grad_(True)

    convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5)).reshape(-1, 1)
    coupling = nn(edge_dist)

    q_source = q[i]
    q_target = q[j]
    pairwise_potential = q_source * q_target * coupling * convergence_func

    results = scatter(
        pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum"
    ).sum(dim=1)

    return results
