from functools import partial

import torch
import torch.nn as nn
import torch_geometric

from e3nn import o3

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin, HeadInterface

from .configs import EScAIPConfigs, init_configs
from .custom_types import GraphAttentionData, GeneralGraphAttentionData
from .modules import (
    EfficientGraphAttentionBlock,
    InputBlock,
    GeneralInputBlock,
    ReadoutBlock,
    OutputProjection,
    OutputLayer,
    CouplingOutputLayer,
    GeneralEfficientGraphAttentionBlock,
)
from .utils.data_preprocess import data_preprocess, data_preprocess_spin_charge
from .utils.nn_utils import no_weight_decay, init_linear_weights
from .utils.graph_utils import (
    unpad_results,
    compilable_scatter,
    potential_full_from_edge_inds,
    heisenberg_potential_full_from_edge_inds,
)
import torch.profiler


@registry.register_model("EScAIP_backbone")
class EScAIPBackbone(nn.Module, GraphModelMixin):
    """
    Efficiently Scaled Attention Interactomic Potential (EScAIP) backbone model.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        cfg = init_configs(EScAIPConfigs, kwargs)
        self.global_cfg = cfg.global_cfg
        self.molecular_graph_cfg = cfg.molecular_graph_cfg
        self.gnn_cfg = cfg.gnn_cfg
        self.reg_cfg = cfg.reg_cfg

        # for trainer
        self.regress_forces = cfg.global_cfg.regress_forces
        self.direct_force = cfg.global_cfg.direct_force
        self.use_pbc = cfg.molecular_graph_cfg.use_pbc

        # graph generation
        self.use_pbc_single = (
            self.molecular_graph_cfg.use_pbc_single
        )  # TODO: remove this when FairChem fixes the bug

        generate_graph_fn = partial(
            self.generate_graph,
            cutoff=self.molecular_graph_cfg.max_radius,
            max_neighbors=self.molecular_graph_cfg.max_neighbors,
            use_pbc=self.molecular_graph_cfg.use_pbc,
            otf_graph=self.molecular_graph_cfg.otf_graph,
            enforce_max_neighbors_strictly=self.molecular_graph_cfg.enforce_max_neighbors_strictly,
            use_pbc_single=self.molecular_graph_cfg.use_pbc_single,
        )

        # data preprocess
        self.data_preprocess = partial(
            data_preprocess,
            generate_graph_fn=generate_graph_fn,
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
        )

        ## Model Components

        # Input Block
        self.input_block = InputBlock(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EfficientGraphAttentionBlock(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers)
            ]
        )

        # Readout Layer
        self.readout_layers = nn.ModuleList(
            [
                ReadoutBlock(
                    global_cfg=self.global_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers + 1)
            ]
        )

        # Output Projection
        self.output_projection = OutputProjection(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # init weights
        self.apply(init_linear_weights)

        # enable torch.set_float32_matmul_precision('high')
        torch.set_float32_matmul_precision("high")

        # log recompiles
        torch._logging.set_logs(recompiles=True)

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def compiled_forward(self, data: GraphAttentionData):
        # input block
        node_features, edge_features = self.input_block(data)

        # input readout
        readouts = self.readout_layers[0](node_features, edge_features)
        node_readouts = [readouts[0]]
        edge_readouts = [readouts[1]]

        # transformer blocks
        for idx in range(self.gnn_cfg.num_layers):
            node_features, edge_features = self.transformer_blocks[idx](
                data, node_features, edge_features
            )
            readouts = self.readout_layers[idx + 1](node_features, edge_features)
            node_readouts.append(readouts[0])
            edge_readouts.append(readouts[1])

        node_features, edge_features = self.output_projection(
            node_readouts=torch.cat(node_readouts, dim=-1),
            edge_readouts=torch.cat(edge_readouts, dim=-1),
        )

        return {
            "data": data,
            "node_features": node_features,
            "edge_features": edge_features,
        }

    @conditional_grad(torch.enable_grad())
    def forward(self, data: torch_geometric.data.Batch):
        # gradient force
        # todo: might need to add global config to specify/detect
        # if self.regress_forces and not self.global_cfg.direct_force:
        data.pos.requires_grad_(True)

        # preprocess data
        x = self.data_preprocess(data)

        return self.forward_fn(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("General_EScAIP_backbone")
class GeneralEScAIPBackbone(nn.Module, GraphModelMixin):
    """
    Efficiently Scaled Attention Interactomic Potential (EScAIP) backbone model.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # load configs
        cfg = init_configs(EScAIPConfigs, kwargs)
        self.global_cfg = cfg.global_cfg
        self.molecular_graph_cfg = cfg.molecular_graph_cfg
        self.gnn_cfg = cfg.gnn_cfg
        self.reg_cfg = cfg.reg_cfg

        # for trainer
        self.regress_forces = cfg.global_cfg.regress_forces
        self.direct_force = cfg.global_cfg.direct_force

        self.use_pbc = cfg.molecular_graph_cfg.use_pbc

        # graph generation
        self.use_pbc_single = (
            self.molecular_graph_cfg.use_pbc_single
        )  # TODO: remove this when FairChem fixes the bug

        generate_graph_fn = partial(
            self.generate_graph,
            cutoff=self.molecular_graph_cfg.max_radius,
            max_neighbors=self.molecular_graph_cfg.max_neighbors,
            use_pbc=self.molecular_graph_cfg.use_pbc,
            otf_graph=self.molecular_graph_cfg.otf_graph,
            enforce_max_neighbors_strictly=self.molecular_graph_cfg.enforce_max_neighbors_strictly,
            use_pbc_single=self.molecular_graph_cfg.use_pbc_single,
        )

        generate_graph_fn_lr = partial(
            self.generate_graph,
            cutoff=self.molecular_graph_cfg.max_radius_lr,
            max_neighbors=self.molecular_graph_cfg.max_neighbors_lr,
            use_pbc=self.molecular_graph_cfg.use_pbc,
            otf_graph=self.molecular_graph_cfg.otf_graph,
            enforce_max_neighbors_strictly=self.molecular_graph_cfg.enforce_max_neighbors_strictly,
            use_pbc_single=self.molecular_graph_cfg.use_pbc_single,
        )

        # data preprocess
        self.data_preprocess = partial(
            data_preprocess_spin_charge,
            generate_graph_fn=generate_graph_fn,
            generate_graph_fn_lr=generate_graph_fn_lr,
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
        )

        ## Model Components

        # Input Block
        self.input_block = GeneralInputBlock(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                GeneralEfficientGraphAttentionBlock(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers)
            ]
        )

        # Readout Layer
        self.readout_layers = nn.ModuleList(
            [
                ReadoutBlock(
                    global_cfg=self.global_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers + 1)
            ]
        )

        # Output Projection
        self.output_projection = OutputProjection(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # init weights
        self.apply(init_linear_weights)

        # enable torch.set_float32_matmul_precision('high')
        torch.set_float32_matmul_precision("high")

        # log recompiles
        torch._logging.set_logs(recompiles=True)

        # print("use compile", self.global_cfg.use_compile)
        # print("atten name", self.gnn_cfg.atten_name)
        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def compiled_forward(self, data: GeneralGraphAttentionData):
        # input block
        node_features, edge_features = self.input_block(data)

        # input readout
        readouts = self.readout_layers[0](node_features, edge_features)
        node_readouts = [readouts[0]]
        edge_readouts = [readouts[1]]

        # transformer blocks
        for idx in range(self.gnn_cfg.num_layers):
            node_features, edge_features = self.transformer_blocks[idx](
                data, node_features, edge_features
            )
            readouts = self.readout_layers[idx + 1](node_features, edge_features)
            node_readouts.append(readouts[0])
            edge_readouts.append(readouts[1])

        node_features, edge_features = self.output_projection(
            node_readouts=torch.cat(node_readouts, dim=-1),
            edge_readouts=torch.cat(edge_readouts, dim=-1),
        )

        return {
            "data": data,
            "node_features": node_features,
            "edge_features": edge_features,
        }

    @conditional_grad(torch.enable_grad())
    def forward(self, data: torch_geometric.data.Batch):
        # gradient force
        # TODO: default this to true for now, update later
        # if (self.regress_forces and not self.global_cfg.direct_force):
        data.pos.requires_grad_(True)

        # preprocess data
        x = self.data_preprocess(data)

        return self.forward_fn(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


class EScAIPHeadBase(nn.Module, HeadInterface):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__()
        self.global_cfg = backbone.global_cfg
        self.molecular_graph_cfg = backbone.molecular_graph_cfg
        self.gnn_cfg = backbone.gnn_cfg
        self.reg_cfg = backbone.reg_cfg
        self.regress_forces = backbone.regress_forces
        self.direct_forces = backbone.direct_force

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("EScAIP_direct_force_head")
class EScAIPDirectForceHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)
        # self.regress_forces = backbone.regress_forces
        # self.direct_forces = backbone.direct_forces

        self.force_direction_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Vector",
        )
        self.force_magnitude_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(self, edge_features, node_features, data: GraphAttentionData):
        # get force direction from edge features
        force_direction = self.force_direction_layer(
            edge_features
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (
            force_direction * data.edge_direction
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (force_direction * data.neighbor_mask.unsqueeze(-1)).sum(
            dim=1
        )  # (num_nodes, 3)
        # get force magnitude from node readouts
        force_magnitude = self.force_magnitude_layer(node_features)  # (num_nodes, 1)
        # get output force
        return force_direction * force_magnitude  # (num_nodes, 3)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            edge_features=emb["edge_features"],
            node_features=emb["node_features"],
            data=emb["data"],
        )

        return unpad_results(
            results={"forces": force_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_energy_head")
class EScAIPEnergyHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)
        # self.regress_forces = backbone.regress_forces
        # self.direct_forces = backbone.direct_forces

        self.energy_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )
        self.energy_reduce = self.gnn_cfg.energy_reduce

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data: GraphAttentionData):
        energy_output = self.energy_layer(node_features)

        # the following not compatible with torch.compile (graph break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        energy_output = compilable_scatter(
            src=energy_output,
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce=self.energy_reduce,
        )
        return energy_output

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )
        return unpad_results(
            results={"energy": energy_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_direct_force_energy_lr_head")
class EScAIPDirectForceEnergyLRHead(EScAIPHeadBase):
    # ONLY CHARGE, takes one dimension - later update for spin charges

    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)

        if self.gnn_cfg.two_component_latent_charge:
            charge_latent_dim = "TwoComp"

            if self.gnn_cfg.heisenberg_tf:
                self.j_coupling_nn = CouplingOutputLayer(
                    global_cfg=self.global_cfg, reg_cfg=self.reg_cfg
                )

        else:
            charge_latent_dim = "Scalar"

        self.q_output_lr = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type=charge_latent_dim,
        )

        self.energy_reduce = self.gnn_cfg.energy_reduce

        # energy terms
        self.energy_layer_sr = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        # force
        self.force_direction_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Vector",
        )

        self.force_magnitude_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.constrain_charge = False
        self.constrain_spin = False
        self.ret_charges = False

        if bool(self.gnn_cfg.constrain_charge):
            self.constrain_charge = True
            self.ret_charges = True

        if bool(self.gnn_cfg.constrain_spin):
            self.constrain_spin = True
            self.ret_spins = True

        # get total charge and regularize with that
        self.post_init(gain=0.01)

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))
        # ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        self.forward_fn = (
            torch.compile(self.compiled_forward, backend="inductor", dynamic=True)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def get_charges(self, node_features, data: GraphAttentionData, unpad: bool = False):
        q_pred = self.q_output_lr(node_features)
        # print("charge shape", q_pred.shape)
        # if two component, square each component to make positive
        # if self.gnn_cfg.two_component_latent_charge:
        #    q_pred = q_pred ** 2

        if unpad:
            q_pred = unpad_results(
                results={"charges": q_pred},
                node_padding_mask=data.node_padding_mask,
                graph_padding_mask=data.graph_padding_mask,
            )["charges"]

        return q_pred

    def get_sr_forces(self, edge_features, node_features, data: GraphAttentionData):
        # get force direction from edge features
        force_direction = self.force_direction_layer(
            edge_features
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (
            force_direction * data.edge_direction
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (force_direction * data.neighbor_mask.unsqueeze(-1)).sum(
            dim=1
        )  # (num_nodes, 3)
        # get force magnitude from node readouts
        force_magnitude = self.force_magnitude_layer(node_features)  # (num_nodes, 1)
        # get output force
        forces_output = force_direction * force_magnitude
        return forces_output

    def get_lr_energies(
        self,
        node_features,
        data: GraphAttentionData,
        return_charges: bool = False,
    ):
        ret_dict = {}

        charges_padded = self.get_charges(node_features, data)

        if self.gnn_cfg.heisenberg_tf:
            energy_spin_raw = heisenberg_potential_full_from_edge_inds(
                pos=data.pos,
                edge_index=data.edge_index_lr,
                q=charges_padded,
                nn=self.j_coupling_nn,
                padding_dim=data.node_padding_mask.shape[0],
            )

            spin_energy_scattered = compilable_scatter(
                src=energy_spin_raw,
                index=data.node_batch,
                dim_size=data.graph_padding_mask.shape[0],
                dim=0,
                reduce="sum",
            )
            # print("scattered: ", spin_energy_scattered.shape)

            if return_charges:
                spin_charges_scattered = compilable_scatter(
                    src=charges_padded,
                    index=data.node_batch,
                    dim_size=data.graph_padding_mask.shape[0],
                    dim=0,
                    reduce="sum",
                )
                # sum energies from alpha and beta
                ret_dict["spin_charges"] = spin_charges_scattered

            ret_dict["energy_spin"] = spin_energy_scattered  # .sum(dim=1)

        if self.gnn_cfg.two_component_latent_charge:
            charges_padded = charges_padded.sum(dim=1)
        else:
            charges_padded = charges_padded

        # charge term
        energy_output_lr = potential_full_from_edge_inds(
            edge_index=data.edge_index_lr,
            pos=data.pos,  # test
            q=charges_padded,
            sigma=1.0,
            epsilon=1e-6,
            padding_dim=data.node_padding_mask.shape[0],
        )

        scattered_energy_lr = compilable_scatter(
            src=energy_output_lr,
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce=self.energy_reduce,
        )
        # print("scattered_energy_lr", scattered_energy_lr.shape)

        if return_charges:
            # print("charges padded: ", charges_padded.shape)
            charges_scattered = compilable_scatter(
                src=charges_padded,
                index=data.node_batch,
                dim_size=data.graph_padding_mask.shape[0],
                dim=0,
                reduce="sum",
            )
            ret_dict["charges"] = charges_scattered

        ret_dict["energy_lr"] = scattered_energy_lr

        return ret_dict

    def compiled_forward(self, edge_features, node_features, data: GraphAttentionData):
        energy_output_sr = self.energy_layer_sr(node_features)

        energy_output_sr = compilable_scatter(
            src=energy_output_sr,
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce=self.energy_reduce,
        )

        forces_output = self.get_sr_forces(
            edge_features=edge_features,
            node_features=node_features,
            data=data,
        )
        energy_lr_dict = self.get_lr_energies(
            node_features=node_features, data=data, return_charges=self.ret_charges
        )

        ret_dict = {
            "energy_sr": energy_output_sr,
            "energy_lr": energy_lr_dict["energy_lr"],
            "forces_sr": forces_output,
        }

        if self.gnn_cfg.heisenberg_tf:
            ret_dict["energy_spin"] = energy_lr_dict["energy_spin"]

        if self.ret_charges:
            # check if this is correct for spin charges
            ret_dict["charges"] = energy_lr_dict["charges"]
            if self.gnn_cfg.two_component_latent_charge:
                ret_dict["spin_charges"] = energy_lr_dict["spin_charges"]

        return ret_dict

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        res_dict_raw = self.forward_fn(
            edge_features=emb["edge_features"],
            node_features=emb["node_features"],
            data=emb["data"],
        )

        # only sr need to be unpadded
        unpad_dict = {
            "energy_sr": res_dict_raw["energy_sr"],
            "forces_sr": res_dict_raw["forces_sr"],
            "energy_lr": res_dict_raw["energy_lr"],
        }

        if self.constrain_charge:
            unpad_dict["charges"] = res_dict_raw["charges"]

        if self.constrain_spin:
            unpad_dict["spin_charges"] = res_dict_raw["spin_charges"]

        if self.gnn_cfg.heisenberg_tf:
            unpad_dict["energy_spin"] = res_dict_raw["energy_spin"]

        res_unpad = unpad_results(
            results=unpad_dict,
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )

        # sum the energies
        energy_output = res_unpad["energy_sr"].view(-1) + res_unpad["energy_lr"].view(
            -1
        )

        if self.gnn_cfg.heisenberg_tf:
            energy_output += res_unpad["energy_spin"].view(-1)

        forces_output = res_unpad["forces_sr"]

        ret_dict = {
            "energy": energy_output,
            "forces": forces_output,
        }

        if self.constrain_charge:
            # if two component, sum the charges for each atom
            if self.gnn_cfg.two_component_latent_charge:
                # print("charges unpadded ",res_unpad["charges"].shape)
                abs_charge = res_unpad["charges"]
                ret_dict["charge"] = abs_charge
            else:
                ret_dict["charge"] = res_unpad["charges"]

        if self.constrain_spin:
            alpha_charge = res_unpad["spin_charges"][:, 0]
            beta_charge = res_unpad["spin_charges"][:, 1]
            spin = alpha_charge - beta_charge
            ret_dict["spin"] = spin

        # print(ret_dict)
        # print(ret_dict.keys())
        return ret_dict


@registry.register_model("EScAIP_grad_force_energy_lr_head")
class EScAIPGradientForceEnergyLRHead(EScAIPHeadBase):
    # ONLY CHARGE, takes one dimension - later update for spin charges

    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)

        # energy terms
        self.energy_layer_sr = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        if self.gnn_cfg.two_component_latent_charge:
            charge_latent_dim = "TwoComp"
        else:
            charge_latent_dim = "Scalar"

        self.q_output_lr = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type=charge_latent_dim,
        )

        self.energy_reduce = self.gnn_cfg.energy_reduce

        # force
        self.force_direction_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Vector",
        )

        self.force_magnitude_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.constrain_charge = False
        self.constrain_spin = False
        self.ret_charges = False
        self.regress_forces = backbone.regress_forces
        self.direct_force = backbone.direct_force

        if bool(self.gnn_cfg.constrain_charge):
            self.constrain_charge = True
            self.ret_charges = True

        if bool(self.gnn_cfg.constrain_spin):
            self.constrain_spin = True
            self.ret_spins = True

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = self.forward

    def get_charges(self, node_features, data: GraphAttentionData, unpad: bool = False):
        q_pred = self.q_output_lr(node_features)
        if unpad:
            q_pred = unpad_results(
                results={"charges": q_pred},
                node_padding_mask=data.node_padding_mask,
                graph_padding_mask=data.graph_padding_mask,
            )["charges"]

        return q_pred

    def get_lr_energies(
        self,
        node_features,
        data: GraphAttentionData,
        return_charges: bool = False,
    ):
        ret_dict = {}

        charges_padded = self.get_charges(node_features, data)

        if self.gnn_cfg.heisenberg_tf:
            energy_spin_raw = heisenberg_potential_full_from_edge_inds(
                pos=data.pos,
                edge_index=data.edge_index_lr,
                q=charges_padded,
                nn=self.coupling_nn,
            )

            spin_energy_scattered = compilable_scatter(
                src=energy_spin_raw,
                index=data.node_batch,
                dim_size=data.graph_padding_mask.shape[0],
                dim=0,
                reduce=self.energy_reduce,
            )

        if self.gnn_cfg.two_component_latent_charge:
            ret_dict["spin_charges"] = charges_padded
            charges_padded = charges_padded.abs().sum(dim=1)
            # print("charges_padded", charges_padded.shape)

        else:
            charges_padded = charges_padded

        energy_output_lr = potential_full_from_edge_inds(
            # edge_dist=data.dist_lr_interactions, # test
            edge_index=data.edge_index_lr,
            # pos=data.pos, # test
            # convergence_func=data.convergence_func, # test
            # edge_dist_transformed=data.edge_dist_transformed, # test
            q=charges_padded,
            sigma=1.0,
            epsilon=1e-6,
            padding_dim=data.node_padding_mask.shape[0],
        )

        scattered_energy_lr = compilable_scatter(
            src=energy_output_lr,
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce=self.energy_reduce,
        )

        if return_charges:
            charges_scattered = compilable_scatter(
                src=charges_padded,
                index=data.node_batch,
                dim_size=data.graph_padding_mask.shape[0],
                dim=0,
                reduce="sum",
            )

            ret_dict["charges"] = charges_scattered

        ret_dict["energy_lr"] = scattered_energy_lr
        if self.gnn_cfg.heisenberg_tf:
            ret_dict["energy_spin"] = spin_energy_scattered

        return ret_dict

    def compiled_forward(self, edge_features, node_features, data: GraphAttentionData):
        energy_output_sr = self.energy_layer_sr(node_features)

        energy_output_sr = compilable_scatter(
            src=energy_output_sr,
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce=self.energy_reduce,
        )

        energy_lr_dict = self.get_lr_energies(
            node_features=node_features,
            data=data,
            return_charges=self.ret_charges,
        )

        # reshape energy lr to n, 1
        energy_lr_dict["energy_lr"] = energy_lr_dict["energy_lr"].view(-1, 1)

        # print("energy_lr_dict", energy_lr_dict["energy_lr"].shape)
        # print("energy_output_sr", energy_output_sr.shape)
        # print("data.pos", data.pos.shape)

        forces_output_sr = (
            -1
            * torch.autograd.grad(
                energy_output_sr.sum(),
                data.pos,
                create_graph=self.training,
                # retain_graph=True,
            )[0]
        )

        forces_output_lr = (
            -1
            * torch.autograd.grad(
                energy_lr_dict["energy_lr"].sum(),
                data.pos,
                create_graph=self.training,
                # retain_graph=True,
            )[0]
        )

        # dummy_tensor = torch.zeros_like(data.pos, requires_grad=True)
        ret_dict = {
            "energy_sr": energy_output_sr,
            "energy_lr": energy_lr_dict["energy_lr"],
            "forces_sr": forces_output_sr,
            "forces_lr": forces_output_lr,
        }

        if self.ret_charges:
            # check if this is correct for spin charges
            ret_dict["charges"] = energy_lr_dict["charges"]  # .view(-1)
            if self.gnn_cfg.two_component_latent_charge:
                ret_dict["spin_charges"] = energy_lr_dict["spin_charges"]

        return ret_dict

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # print("HEHE ** " * 10)
        res_dict_raw = self.compiled_forward(
            edge_features=emb["edge_features"],
            node_features=emb["node_features"],
            data=emb["data"],
        )

        # print("charges", energy_lr_dict["charges"].shape)

        # only sr need to be unpadded
        unpad_dict = {
            "energy_sr": res_dict_raw["energy_sr"],
            # "forces_sr": res_dict_raw["forces_sr"],
            "energy_lr": res_dict_raw["energy_lr"],
            # "forces_lr": res_dict_raw["forces_lr"],
        }

        if self.constrain_charge:
            unpad_dict["charges"] = res_dict_raw["charges"]
        if self.constrain_spin:
            unpad_dict["spin_charges"] = res_dict_raw["spin_charges"]

        res_unpad = unpad_results(
            results=unpad_dict,
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )

        # sum the energies
        energy_output = res_unpad["energy_sr"].view(-1) + res_unpad["energy_lr"].view(
            -1
        )
        forces_output = res_dict_raw["forces_sr"] + res_dict_raw["forces_lr"]

        # yells at you if you don't use all of these terms for loss calc
        ret_dict = {
            "energy": energy_output,
            "forces": forces_output,
        }

        if self.constrain_charge:
            # if two component, sum the charges for each atom
            if self.gnn_cfg.two_component_latent_charge:
                # print("charges unpadded ",res_unpad["charges"].shape)
                abs_charge = res_unpad["charges"]
                ret_dict["charge"] = abs_charge
            else:
                ret_dict["charge"] = res_unpad["charges"]

        if self.constrain_spin:
            alpha_charge = res_unpad["spin_charges"][:, 0]
            beta_charge = res_unpad["spin_charges"][:, 1]
            spin = alpha_charge - beta_charge
            ret_dict["spin"] = spin

        return ret_dict


@registry.register_model("EScAIP_grad_energy_force_head")
class EScAIPGradientEnergyForceHead(EScAIPEnergyHead):
    """
    Do not support torch.compile
    """

    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.energy_layer(emb["node_features"])

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

        energy_output = compilable_scatter(
            src=energy_output,
            index=emb["data"].node_batch,
            dim_size=emb["data"].graph_padding_mask.shape[0],
            dim=0,
            reduce=self.energy_reduce,
        )

        forces_output = (
            -1
            * torch.autograd.grad(
                energy_output.sum(),
                data.pos,
                create_graph=self.training,
                # retain_graph=True,
            )[0]
        )

        return unpad_results(
            results={"energy": energy_output, "forces": forces_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_rank2_head")
class EScAIPRank2Head(EScAIPHeadBase):
    """
    Rank-2 head for EScAIP model. Modified from the Rank2Block for Equiformer V2.
    """

    def __init__(
        self,
        backbone: EScAIPBackbone,
        output_name: str = "stress",
    ):
        super().__init__(backbone)
        self.output_name = output_name
        self.scalar_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )
        self.irreps2_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )
        # self.regress_forces = backbone.regress_forces
        # self.direct_forces = backbone.direct_forces

        self.post_init()

    def compiled_forward(self, node_features, edge_features, data: GraphAttentionData):
        sphere_irrep2 = o3.spherical_harmonics(
            2, data.edge_direction, True
        ).detach()  # (num_nodes, max_neighbor, 5)

        # map from invariant to irrep2
        edge_irrep2 = (
            sphere_irrep2[:, :, :, None] * edge_features[:, :, None, :]
        )  # (num_nodes, max_neighbor, 5, h)

        # sum over neighbors
        neighbor_count = data.neighbor_mask.sum(dim=1, keepdim=True) + 1e-5
        neighbor_count = neighbor_count.to(edge_irrep2.dtype)
        node_irrep2 = (
            edge_irrep2 * data.neighbor_mask.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1) / neighbor_count.unsqueeze(-1)  # (num_nodes, 5, h)

        irrep2_output = self.irreps2_layer(node_irrep2)  # (num_nodes, 5, 1)
        scalar_output = self.scalar_layer(node_features)  # (num_nodes, 1)

        # get graph level output
        irrep2_output = compilable_scatter(
            src=irrep2_output.view(-1, 5),
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )
        scalar_output = compilable_scatter(
            src=scalar_output.view(-1, 1),
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )
        return irrep2_output, scalar_output.view(-1)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        irrep2_output, scalar_output = self.forward_fn(
            node_features=emb["node_features"],
            edge_features=emb["edge_features"],
            data=emb["data"],
        )
        output = {
            f"{self.output_name}_isotropic": scalar_output.unsqueeze(1),
            f"{self.output_name}_anisotropic": irrep2_output,
        }

        return unpad_results(
            results=output,
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )
