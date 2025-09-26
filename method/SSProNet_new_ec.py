from torch_geometric.nn import inits, MessagePassing
from torch_geometric.nn import radius_graph

from .features import d_angle_emb, d_theta_phi_emb

from torch_scatter import scatter 
from torch_sparse import matmul

import torch 
from torch import nn
from torch.nn import Embedding 
import torch.nn.functional as F

import numpy as np

import os, sys

# Add parent folder (ProNet) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


num_aa_type = 26
num_side_chain_embs = 8
num_bb_embs = 6

def swish(x): 
    return x * torch.sigmoid(x)

class Linear(torch.nn.Module):
    """
        A linear method encapsulation similar to PyG's

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
        bias (int)
        weight_initializer (string): (glorot or zeros)
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer='glorot'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'zeros':
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)

class TwoLinear(torch.nn.Module):
    """
        A layer with two linear modules

        Parameters
        ----------
        in_channels (int)
        middle_channels (int)
        out_channels (int)
        bias (bool)
        act (bool)
    """

    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False
    ):
        super(TwoLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x
    
class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l =Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight, size=None): 
       x =(x,x)
       out =self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
       out = self.lin_l(out)
       return out+ self.lin_r(x[1])
    
    def message(self, x_j,edge_weight):
  
        return edge_weight*x_j
    
    def message_and_aggregate(self, adj_t, x):
     
        return matmul(adj_t, x[0], reduce=self.aggr)



class InteractionBlock(torch.nn.Module):

    def __init__(self, hidden_channels, output_channels, num_radical, num_spherical, num_layers, mid_emb, act=swish,
                 num_pos_emb=16, dropout=0.0, level='allatom'): 

        super(InteractionBlock,self).__init__()
        self.act = act 
        self.dropout = nn.Dropout(dropout)

        self.conv0=EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1=EdgeGraphConv(hidden_channels, hidden_channels) 
        self.conv2=EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin_feature0 = TwoLinear(num_radical*num_spherical**2, mid_emb, hidden_channels)
        if level =='aminoacid':
            self.lin_feature1 = TwoLinear(num_radical*num_spherical, mid_emb, hidden_channels)
        if level =='backbone' or level =='allatom':
            self.lin_feature1 = TwoLinear(3*num_radical*num_spherical, mid_emb, hidden_channels) 
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels) 

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, output_channels)    

        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = torch.nn.ModuleList()
        self.lin_cat.append(Linear(3*hidden_channels, hidden_channels))
        for _ in range(num_layers-1):
            self.lin_cat.append(Linear(hidden_channels, hidden_channels))
        
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))

        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for lin  in self.lin_cat:
            lin.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.final.reset_parameters()

    def forward(self, x, feature0, feature1, pos_emb, edge_index, batch):
        # x.shape = [batch_size,hidden_channels]
        x_lin_1 = self.act(self.lin_1(x)) # x_lin_1.shape = [batch_size,hidden_channels]
        x_lin_2 = self.act(self.lin_2(x)) # x_lin_2.shape = [batch_size,hidden_channels]

        # feature0.shape = [nbre_edges, num_radical*num_spherical**2]
        feature0 = self.lin_feature0(feature0) # feature0.shape = [nbre_edges, hidden_channels]
        h0 = self.conv0(x_lin_1, edge_index, feature0) # h0.shape = [batch_size,hidden_channels]
        h0 = self.lin0(h0) # h0.shape = [batch_size,hidden_channels]
        h0 = self.act(h0) # h0.shape = [batch_size,hidden_channels]
        h0 = self.dropout(h0) # h0.shape = [batch_size,hidden_channels]

        # feature1.shape = [nbre_edges, num_radical*num_spherical] or [nbre_edges, 3*num_radical*num_spherical]
        feature1 = self.lin_feature1(feature1) # feature1.shape = [nbre_edges ,hidden_channels]
        h1 = self.conv1(x_lin_1,edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        # pos_emb.shape = [num_pos_emb]
        feature2 = self.lin_feature2(pos_emb)   # feature2.shape = [batch_size, hidden_channels]
        h2 = self.conv2(x_lin_1, edge_index, feature2) # h2.shape = [batch_size,hidden_channels]
        h2 = self.lin2(h2) # h2.shape = [batch_size,hidden_channels]
        h2 = self.act(h2) # h2.shape = [batch_size,hidden_channels]
        h2 = self.dropout(h2) # h2.shape = [batch_size,hidden_channels] 

        h = torch.cat((h0,h1,h2),dim=1)

        for lin in self.lin_cat:
            h = self.act(lin(h))# h.shape = [batch_size,hidden_channels]
        # h.shape = [batch_size,hidden_channels]
        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(lin(h))

        h = self.final(h)# h.shape = [batch_size,output_channels]
        return h       


class SSProNet(nn.Module):

    def __init__(
        self,
        level='aminoacid',
        num_blocks=4,
        hidden_channels=128,
        out_channels=1,
        mid_emb=64,
        num_radial=6,
        num_spherical=2,
        cutoff=10.0,
        max_num_neighbors=32,
        int_emb_layers=3,
        out_layers=2,
        num_pos_emb=16,
        dropout=0,
        data_augment_eachlayer=False,
        euler_noise=False,
    ):
        super(SSProNet, self).__init__()  # <-- fix

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_pos_emb = num_pos_emb
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.level = level
        self.act = swish

        # geometric edge feature builders (same as ProNet)
        self.feature0 = d_theta_phi_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature1 = d_angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        # ----- unified raw->hidden embedding dims -----
        aa_dim = num_aa_type                 # 26 (one-hot)
        ss_dim = 8                           # DSSP secondary structure one-hot
        acc_dim = 1                          # solvent accessibility scalar

        if level == 'aminoacid':
            input_dim = aa_dim + ss_dim + acc_dim
            self.embedding = nn.Linear(input_dim, hidden_channels)

        elif level == 'backbone':
            input_dim = aa_dim + num_bb_embs + ss_dim + acc_dim  # 26 + 6 + 8 + 1 = 41
            self.embedding = nn.Linear(input_dim, hidden_channels)

        elif level == 'allatom':
            input_dim = aa_dim + num_bb_embs + num_side_chain_embs + ss_dim + acc_dim  # 26 + 6 + 8 + 8 + 1 = 49
            self.embedding = nn.Linear(input_dim, hidden_channels)

        else:
            raise ValueError(f"Unsupported level: {level}")

        # interaction blocks (unchanged signature for now; you will extend later)
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(
                hidden_channels=hidden_channels,
                output_channels=hidden_channels,
                num_radical=num_radial,
                num_spherical=num_spherical,
                num_layers=int_emb_layers,
                mid_emb=mid_emb,
                act=self.act,
                num_pos_emb=num_pos_emb,
                dropout=dropout,
                level=level,
            ) for _ in range(num_blocks)
        ])

        self.lins_out = nn.ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(out_layers-1)])
        self.lin_out = Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins_out:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def pos_emb(self,edge_index, num_pos_emb=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=d.device) *
            - (np.log(10000.0) / num_pos_emb)
        )

        angles = d.unsqueeze(-1)*frequency
        E =  torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        return E
    

    def extract_hbond_edges_from_data(self, data, energy_threshold: float = -0.5, add_reverse: bool = True):
        """
        Build hydrogen-bond edge_index from data.hbonds.

        Parameters
        ----------
        energy_threshold : float
            Only keep bonds with energy <= threshold (default: -0.5).
            More negative = stronger bond.
        add_reverse : bool
            If True, add both directions (make undirected).
        """
        device = data.coords_ca.device
        hbonds = getattr(data, "hbonds", None)
        rows, cols = [], []

        def _add(src, tgt, energy):
            if energy <= energy_threshold:         # <-- filter here
                rows.append(int(src))
                cols.append(int(tgt))
                if add_reverse:
                    rows.append(int(tgt))
                    cols.append(int(src))

        if hbonds is None:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        ptr = getattr(data, "ptr", None)
        if isinstance(hbonds, list) and len(hbonds) > 0 and isinstance(hbonds[0], list) and ptr is not None:
            # batched graphs
            num_graphs = ptr.numel() - 1
            for g in range(num_graphs):
                off = int(ptr[g].item())
                for bond in hbonds[g]:
                    if isinstance(bond, (list, tuple)) and len(bond) >= 2:
                        src, tgt = int(bond[0]) + off, int(bond[1]) + off
                        energy = float(bond[2]) if len(bond) >= 3 else 0.0
                        _add(src, tgt, energy)
        elif isinstance(hbonds, list):
            for bond in hbonds:
                if isinstance(bond, (list, tuple)) and len(bond) >= 2:
                    src, tgt = int(bond[0]), int(bond[1])
                    energy = float(bond[2]) if len(bond) >= 3 else 0.0
                    _add(src, tgt, energy)
        elif torch.is_tensor(hbonds) and hbonds.dim() == 2 and hbonds.size(1) >= 2:
            srcs, tgts = hbonds[:, 0].tolist(), hbonds[:, 1].tolist()
            energies = hbonds[:, 2].tolist() if hbonds.size(1) >= 3 else [0.0] * hbonds.size(0)
            for src, tgt, energy in zip(srcs, tgts, energies):
                _add(src, tgt, energy)

        if len(rows) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        return torch.tensor([rows, cols], dtype=torch.long, device=device)



    def forward(self, batch_data):
    
        z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch 
        pos_n = batch_data.coords_n
        pos_c = batch_data.coords_c
        bb_embs = batch_data.bb_embs
        side_chain_embs = batch_data.side_chain_embs
        ss = batch_data.ss.long()                        # [nbre_nodes]  values in {0..7}
        acc = batch_data.acc.float()                     # [nbre_nodes]  solvent accessibility (scalar)


        device = z.device

        # ----- node features: concat raw -> embed -----
        aa_onehot = F.one_hot(z, num_classes=num_aa_type).float()          # [nbre_nodes, 26]
        ss_onehot = F.one_hot(ss, num_classes=8).float()                   # [nbr_nodes, 8]
        acc_col = acc.unsqueeze(-1)                                        # [nbre_nodes, 1]

        if self.level == 'aminoacid':
            feats = torch.cat([aa_onehot, ss_onehot, acc_col], dim=1)      # [nbre_nodes, 26+8+1]
        elif self.level == 'backbone':
            feats = torch.cat([aa_onehot, bb_embs, ss_onehot, acc_col], dim=1)
        elif self.level == 'allatom':
            feats = torch.cat([aa_onehot, bb_embs, side_chain_embs, ss_onehot, acc_col], dim=1)
        else:
            raise ValueError(f"Unsupported level: {self.level}")

        x = self.embedding(feats)  # [nbre_nodes, hidden_channels]
            

        # 2) Build both edge sets

        edge_index_radius = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )  # [2, nbr_edges_RADIUS]

        edge_index_hbond = self.extract_hbond_edges_from_data(batch_data)  # [2, Eh]
    
        # 3) Merge → de-duplicate → remove self-loops
        if edge_index_hbond.numel() > 0:
            edge_index = torch.cat([edge_index_radius, edge_index_hbond], dim=1)
        else:
            edge_index = edge_index_radius

        # drop identical columns
        edge_index = torch.unique(edge_index, dim=1)

        # drop self-loops (just in case)
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]


        pos_emb = self.pos_emb(edge_index, self.num_pos_emb) # pos_emb.shape = [num_edges_RADIUS, num_pos_emb]
        j, i = edge_index



        # Calculate distances.
        dist = (pos[i] - pos[j]).norm(dim=1)  # dist.shape = [num_edges_RADIUS]

        num_nodes = len(z)

        # Calculate angles theta and phi.
        refi0 = (i-1)%num_nodes
        refi1 = (i+1)%num_nodes

        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i], dim=-1).norm(dim=-1)
        theta = torch.atan2(b, a) # theta.shape = [num_edges_RADIUS]

        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i], dim=-1)
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i], dim=-1)
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2,dim=-1) * (pos[refi0] - pos[i])).sum(dim=-1) / ((pos[refi0] - pos[i]).norm(dim=-1))
        phi = torch.atan2(b, a) # phi.shape = [num_edges_RADIUS]

        feature0 = self.feature0(dist, theta, phi) # feature0.shape = [num_edges_RADIUS, num_radical*num_spherical**2]

        if self.level == 'backbone' or self.level == 'allatom':
            # Calculate Euler angles.
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i], dim=-1), dim=-1)
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7
            
            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j], dim=-1), dim=-1)
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7

            Or1_Or2_N = torch.cross(Or1_z, Or2_z, dim=-1)
            
            angle1 = torch.atan2((torch.cross(Or1_x, Or1_Or2_N, dim=-1) * Or1_z).sum(dim=-1)/Or1_z_length, (Or1_x * Or1_Or2_N).sum(dim=-1))
            angle2 = torch.atan2(torch.cross(Or1_z, Or2_z, dim=-1).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1))
            angle3 = torch.atan2((torch.cross(Or1_Or2_N, Or2_x, dim=-1) * Or2_z).sum(dim=-1)/Or2_z_length, (Or1_Or2_N * Or2_x).sum(dim=-1))

            if self.euler_noise:
                euler_noise = torch.clip(torch.empty(3,len(angle1)).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]
            feature1 = torch.cat((self.feature1(dist, angle1), self.feature1(dist, angle2), self.feature1(dist, angle3)),1) # feature1.shape = [num_edges_RADIUS, 3*num_radical*num_spherical]

        elif self.level == 'aminoacid':
            refi = (i-1)%num_nodes

            refj0 = (j-1)%num_nodes
            refj = (j-1)%num_nodes
            refj1 = (j+1)%num_nodes

            mask = refi0 == j
            refi[mask] = refi1[mask]
            mask = refj0 == i
            refj[mask] = refj1[mask]

            plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i], dim=-1)
            plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j], dim=-1)
            a = (plane1 * plane2).sum(dim=-1) 
            b = (torch.cross(plane1, plane2, dim=-1) * (pos[j] - pos[i])).sum(dim=-1) / dist
            tau = torch.atan2(b, a)

            feature1 = self.feature1(dist, tau) # feature1.shape = [num_edges_RADIUS, num_radical*num_spherical]
        

        # 5) Run original ProNet blocks with the unified edge_index
        for block in self.interaction_blocks:
            if self.data_augment_eachlayer:
                x = x + torch.empty_like(x).normal_(0.0, 0.025).clamp_(-0.1, 0.1)
            x = block(x, feature0, feature1, pos_emb, edge_index, batch)  # <-- ProNet call


        #  x.shape: [batch_size,hidden_channels]
        y = scatter(x, batch, dim=0) # pools nodes of the same graph in the batch into a single node
        # y.shape = [nbr_of_graphs_in_batch_data,hidden_channels]
        # e.g. : if batch contains 2 graphs, then y.shape = [2,hidden_channels]

        for lin in self.lins_out:
            y = self.relu(lin(y))
            y = self.dropout(y)  

        y = self.lin_out(y) # y.shape = [nbr_graphs_in_batch_data, out_channels]
        return y
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())



def TEST_SSProNet(level="allatom", out_channels=3):
    """
    Test the SSProNet model with toy protein graphs.
    """
    import torch
    import numpy as np
    from torch_geometric.data import Data, Batch

    num_aa_types = 26
    num_bb_embs = 6
    num_side_chain_embs = 8

    # --- Toy protein sequences ---
    protein_seq1 = "ACDFG"
    protein_seq2 = "DGAAC"
    aa_types = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    aa_to_id = {aa: idx for idx, aa in enumerate(aa_types)}

    def sequence_to_indices(sequence: str) -> torch.Tensor:
        indices = [aa_to_id[aa] for aa in sequence]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(1)

    x1 = sequence_to_indices(protein_seq1)  # [5,1]
    x2 = sequence_to_indices(protein_seq2)  # [5,1]

    # Random 3D coords
    coords_ca_1 = np.random.rand(len(protein_seq1), 3).astype(np.float32)
    coords_ca_2 = np.random.rand(len(protein_seq2), 3).astype(np.float32)
    coords_n_1 = np.random.rand(len(protein_seq1), 3).astype(np.float32)
    coords_n_2 = np.random.rand(len(protein_seq2), 3).astype(np.float32)
    coords_c_1 = np.random.rand(len(protein_seq1), 3).astype(np.float32)
    coords_c_2 = np.random.rand(len(protein_seq2), 3).astype(np.float32)

    # Random bb & sidechain
    bb_embs_1 = np.random.rand(len(protein_seq1), num_bb_embs).astype(np.float32)
    bb_embs_2 = np.random.rand(len(protein_seq2), num_bb_embs).astype(np.float32)
    side_chain_embs_1 = np.random.rand(len(protein_seq1), num_side_chain_embs).astype(np.float32)
    side_chain_embs_2 = np.random.rand(len(protein_seq2), num_side_chain_embs).astype(np.float32)

    # --- DSSP mock features ---
    ss1 = torch.randint(0, 8, (len(protein_seq1),))       # [5] secondary structure labels
    ss2 = torch.randint(0, 8, (len(protein_seq2),))       # [5]
    acc1 = torch.rand(len(protein_seq1))                  # [5] solvent accessibility
    acc2 = torch.rand(len(protein_seq2))                  # [5]

    # Example HB edges: (src, tgt, energy)
    hbonds1 = [(0, 2, -1.2), (3, 4, -0.5)]
    hbonds2 = [(1, 3, -0.9)]

    # Build PyG Data objects
    data_1 = Data(
        x=x1,
        coords_ca=torch.tensor(coords_ca_1),
        coords_c=torch.tensor(coords_c_1),
        coords_n=torch.tensor(coords_n_1),
        bb_embs=torch.tensor(bb_embs_1),
        side_chain_embs=torch.tensor(side_chain_embs_1),
        ss=ss1,
        acc=acc1,
        hbonds=hbonds1,
    )

    data_2 = Data(
        x=x2,
        coords_ca=torch.tensor(coords_ca_2),
        coords_c=torch.tensor(coords_c_2),
        coords_n=torch.tensor(coords_n_2),
        bb_embs=torch.tensor(bb_embs_2),
        side_chain_embs=torch.tensor(side_chain_embs_2),
        ss=ss2,
        acc=acc2,
        hbonds=hbonds2,
    )

    # Batch both proteins
    batch = Batch.from_data_list([data_1, data_2])

    print("Batch summary:")
    print("batch.x.shape =", batch.x.shape)
    print("batch.coords_ca.shape =", batch.coords_ca.shape)
    print("batch.bb_embs.shape =", batch.bb_embs.shape)
    print("batch.side_chain_embs.shape =", batch.side_chain_embs.shape)
    print("batch.ss.shape =", batch.ss.shape)
    print("batch.acc.shape =", batch.acc.shape)
    print("batch.hbonds =", batch.hbonds)
    print("Number of HB edges in graph1 =", len(data_1.hbonds))
    print("Number of HB edges in graph2 =", len(data_2.hbonds))

    # --- Initialize SSProNet ---
    model = SSProNet(
        level=level,
        num_blocks=2,
        hidden_channels=16,
        out_channels=out_channels,
        mid_emb=32,
        num_radial=6,
        num_spherical=2,
        cutoff=10.0,
        max_num_neighbors=8,
        int_emb_layers=2,
        out_layers=2,
        num_pos_emb=8,
        dropout=0.1,
    )

    out = model(batch)  # [num_graphs, out_channels]
    print("\n--- Test SSProNet ---")
    print("level =", level)
    print("Output shape =", out.shape)   # Expected: [2, out_channels]
    print("Output values =", out)
    print("Num params =", model.num_params)


if __name__  == '__main__':
   
   TEST_SSProNet(level="aminoacid", out_channels=3)

   pass



