import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .base import BaseModel
from .gvp import GVP, GVPConvLayer, LayerNorm
from .schnet import SchNet
from .attention import MultiHeadAttention
from .interaction import EnzymeCompoundCrossAttention, CrossAttention


def get_dis_pair(coords, bin_size=2, bin_min=-1, bin_max=30, num_classes=16):
    dis_pair = torch.cdist(coords, coords, compute_mode='donot_use_mm_for_euclid_dist')
    dis_pair[dis_pair>bin_max] = bin_max
    dis_pair_bin_index = torch.div(dis_pair - bin_min, bin_size, rounding_mode='floor').long()
    dis_pair_bin = torch.nn.functional.one_hot(dis_pair_bin_index, num_classes=num_classes)
    dis_pair_bin = dis_pair_bin.float()
    return dis_pair_bin, dis_pair


def calculate_pocket_weights(enzyme_coords_batch, enzyme_coords_mask):
    # 计算几何中心
    # print(f"enzyme_coords_batch.shape: {enzyme_coords_batch.shape}, enzyme_coords_mask.shape: {enzyme_coords_mask.shape}")
    valid_coords = enzyme_coords_batch
    valid_counts = enzyme_coords_mask.sum(dim=1, keepdim=True).float()  # (bs, 1)

    # 避免除以0的情况
    valid_counts[valid_counts == 0] = 1

    # 计算几何中心坐标
    center_of_mass = valid_coords.sum(dim=1) / valid_counts  # (bs, 3)

    # 计算每个氨基酸到几何中心的距离
    distances = torch.norm(enzyme_coords_batch - center_of_mass.unsqueeze(1), dim=2)  # (bs, n_nodes)

    # 计算权重（距离越小权重越大）
    max_distances, _ = torch.max(distances, dim=1, keepdim=True)  # (bs, 1)
    min_distances, _ = torch.min(distances, dim=1, keepdim=True)  # (bs, 1)

    # 避免除以0
    distance_range = max_distances - min_distances
    distance_range[distance_range == 0] = 1

    # 归一化的距离
    normalized_distances = (distances - min_distances) / distance_range  # (bs, n_nodes)

    # 计算权重（权重越小表示距离几何中心越远），设置范围为[0.75, 1]
    pocket_weight = (1 - normalized_distances) / 5  # (bs, n_nodes)

    # 对于无效氨基酸，权重设为0
    pocket_weight = pocket_weight * enzyme_coords_mask

    return pocket_weight


def calc_interaction_weight(enzyme_coords_batch, enzyme_coords_mask, substrate_reacting_center, substrate_mask):
    
    pocket_weight = calculate_pocket_weights(enzyme_coords_batch, enzyme_coords_mask)
    
    # from one-hot to new value range: [1, 2]
    substrate_weight = (substrate_reacting_center * 0.5 + 0.1) * substrate_mask
    
    interaction_weight = torch.einsum('bi,bj->bij', substrate_weight, pocket_weight)
    
    return interaction_weight


class GVP_embedding(nn.Module):

    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):

        super(GVP_embedding, self).__init__()
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (2*ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq=None):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if seq:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        return out


class EnzymeCAGE(BaseModel):
    def __init__(
        self, 
        use_esm=True, 
        use_structure=True, 
        use_drfp=True, 
        use_prods_info=True, 
        esm_model=None,
        interaction_method='geo-enhanced-interaction',
        rxn_inner_interaction=True,
        pocket_inner_interaction=True,
        hidden_dims=[3840, 2048, 1024], 
        dropout=0.2, 
        sigmoid_readout=False,
        device='cpu'
        ):
        super(EnzymeCAGE, self).__init__()
        self.use_esm = use_esm
        self.use_structure = use_structure
        self.use_drfp = use_drfp
        self.use_prods_info = use_prods_info
        self.interaction_method = interaction_method
        self.rxn_inner_interaction = rxn_inner_interaction
        self.pocket_inner_interaction = pocket_inner_interaction
        self.embed_dim = 128
        self.pair_repr_dim = 32
        self.attention_output_dim = 128
        self.model_device = device
        self.sigmoid_readout = sigmoid_readout
        self.dis_onehot_class = 16
        
        if esm_model == 'ESM-C_600M':
            self.esm_dim = 1152
        elif esm_model == 'esm2_t33_650M_UR50D' or not esm_model:
            self.esm_dim = 1280
        else:
            raise ValueError('Invalid ESM model name')

        in_feat_dim = 0
        if self.use_esm:
            in_feat_dim += self.esm_dim
        if self.use_structure:
            if self.interaction_method == 'geo-enhanced-interaction':
                in_feat_dim += self.attention_output_dim * 4
            elif self.interaction_method == 'no_interaction':
                in_feat_dim += self.embed_dim * 3

        # reaction feature
        if self.use_drfp:
            in_feat_dim += 2048
        
        if not self.use_prods_info:
            print('Ignore product information of reaction!')
            if self.interaction_method == 'geo-enhanced-interaction':
                in_feat_dim -= self.attention_output_dim * 2
            elif self.interaction_method == 'no_interaction':
                in_feat_dim -= self.embed_dim

        self.molecule_encoder = SchNet(hidden_channels=self.embed_dim).to(device)
        self.mol_repr_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.gvp_encoder = GVP_embedding((6, 3), (self.embed_dim//2, 16), (32, 1), (32, 1), seq_in=False)
        self.enzyme_transform_layer = nn.Linear(self.esm_dim + self.embed_dim, self.embed_dim)
        
        self.pair_repr_linear = nn.Linear(self.dis_onehot_class, 32)

        if self.interaction_method == 'geo-enhanced-interaction':
            enz_node_dim = self.esm_dim + self.embed_dim
            enzyme_attn_embed_dim = 512
            self.enzyme_attention = MultiHeadAttention(num_heads=8, embed_dim=enzyme_attn_embed_dim, input_dim=enz_node_dim)
            self.substrate_attention = MultiHeadAttention(num_heads=8, embed_dim=self.embed_dim, input_dim=self.embed_dim)
            
            if self.rxn_inner_interaction:
                self.reaction_cross_attn = CrossAttention(self.embed_dim, self.embed_dim, self.embed_dim)
            
            self.interaction_model = EnzymeCompoundCrossAttention(enzyme_attn_embed_dim, self.embed_dim, self.attention_output_dim, self.use_prods_info)
            
        elif self.interaction_method == 'no_interaction':
            pass
        else:
            raise ValueError(f'infomation fusion method not supported: {self.interaction_method}')
        
        print(f'in_feat_dim: {in_feat_dim}')
        assert in_feat_dim > 0
        hidden_dims[0] = in_feat_dim
        
        self.dropout = nn.Dropout2d(p=dropout)
        
        layer1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU()
        )
        layer2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU()
        )
        layer3 = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 1),
        )
        self.mlp = nn.Sequential(layer1, layer2, layer3)
    
    def encode_molecule(self, data):
        atom_feats_left, atom_feats_right = data['substrates'].x, data['products'].x
        coords_3d_left, coords_3d_right = data['substrates'].positions, data['products'].positions
        batch_index_left, batch_index_right = data['substrates'].batch, data['products'].batch

        _, substrates_repr = self.molecule_encoder(atom_feats_left[:, 0], coords_3d_left, batch_index_left, return_latent=True)
        _, products_repr = self.molecule_encoder(atom_feats_right[:, 0], coords_3d_right, batch_index_right, return_latent=True)
        
        return substrates_repr, products_repr
    
    def forward(self, data):
        all_features = []
                
        if self.use_structure:
            nodes = (data['protein']['node_s'], data['protein']['node_v'])
            edges = (data[("protein", "p2p", "protein")]["edge_s"], data[("protein", "p2p", "protein")]["edge_v"])
            enzyme_batch = data['protein'].batch
            gvp_output = self.gvp_encoder(nodes, data[("protein", "p2p", "protein")]["edge_index"], edges)
                        
            esm_output = data['protein'].esm_node_feature
            enzyme_output = torch.cat([gvp_output, esm_output], dim=-1)
            enzyme_out_batched, enzyme_out_mask = to_dense_batch(enzyme_output, enzyme_batch)
            
            substrates_repr, products_repr = self.encode_molecule(data)
            subs_repr_batched, subs_repr_mask = to_dense_batch(substrates_repr, data['substrates'].batch)
            prods_repr_batched, prods_repr_mask = to_dense_batch(products_repr, data['products'].batch)
            
            
            if self.interaction_method == 'geo-enhanced-interaction':
                
                # Geometry attention part
                p_coords_batched, p_coords_mask = to_dense_batch(data.node_xyz, data['protein'].batch)
                _, protein_dis_pair = get_dis_pair(p_coords_batched, bin_size=2, bin_min=-1, bin_max=30, num_classes=self.dis_onehot_class)
                
                if self.pocket_inner_interaction:
                    protein_attn_bias = 1 - protein_dis_pair / 30
                else:
                    protein_attn_bias = None
                enzyme_out_batched, _ = self.enzyme_attention(enzyme_out_batched, enzyme_out_batched, enzyme_out_mask, attn_bias=protein_attn_bias, return_weights=True)
                
                subs_reacting_center, subs_mask = to_dense_batch(data['substrates'].reacting_center, data['substrates'].batch)
                prods_reacting_center, prods_mask = to_dense_batch(data['products'].reacting_center, data['products'].batch)
                
                if self.rxn_inner_interaction:
                    substrate_weight = (subs_reacting_center * 0.5 + 0.1) * subs_mask
                    product_weight = (prods_reacting_center * 0.5 + 0.1) * prods_mask
                    rxn_interaction_weight = torch.einsum('bi,bj->bij', substrate_weight, product_weight)
                    subs_repr_batched, _ = self.reaction_cross_attn(subs_repr_batched, prods_repr_batched, prods_repr_batched, 
                                                                    subs_repr_mask, prods_repr_mask, rxn_interaction_weight)
                    
                    if self.use_prods_info:
                        prods_repr_batched, _ = self.reaction_cross_attn(prods_repr_batched, subs_repr_batched, subs_repr_batched, 
                                                                        prods_repr_mask, subs_repr_mask, rxn_interaction_weight.transpose(1, 2))
                    
                interaction_weight = calc_interaction_weight(p_coords_batched, p_coords_mask, subs_reacting_center, subs_mask)
                
                infomation_fused, _ = self.interaction_model(enz_node_feature=enzyme_out_batched,
                                   substrate_node_feature=subs_repr_batched,
                                   product_node_feature=prods_repr_batched,
                                   enz_node_feature_mask=enzyme_out_mask,
                                   substrate_node_feature_mask=subs_repr_mask,
                                   product_node_feature_mask=prods_repr_mask,
                                   interaction_weight=interaction_weight,
                                   return_weights=True)
                
            elif self.interaction_method == 'no_interaction':
                enzyme_out_batched = self.enzyme_transform_layer(enzyme_out_batched)
                infomation_fused = torch.cat([subs_repr_batched.mean(dim=1), enzyme_out_batched.mean(dim=1)], dim=1)

            else:
                raise ValueError(f'infomation fusion method not supported: {self.interaction_method}')
            
            all_features.append(infomation_fused)
            
        if self.use_esm:
            esm_embedding, _ = to_dense_batch(data.esm_feature, data.esm_feature_batch)
            all_features.append(esm_embedding)
        
        if self.use_drfp:
            reaction_feature, _ = to_dense_batch(data.reaction_feature, data.reaction_feature_batch)
            all_features.append(reaction_feature)

        # for each in all_features:
        #     print(each.shape)
        
        all_features = torch.cat(all_features, dim=-1)
        output = self.mlp(all_features).squeeze(-1)
        if self.sigmoid_readout:
            output = torch.sigmoid(output)

        return output
    
    @torch.no_grad()
    def predict(self, dataloader):
        preds = []
        for batch in dataloader:
            pred = self.forward(batch.to(self.model_device))
            preds.append(pred)
        pred = torch.cat(preds)
        
        if not self.sigmoid_readout:
            pred = torch.sigmoid(pred)
        
        return pred
