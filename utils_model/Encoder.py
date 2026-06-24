import torch
import torch.nn as nn
import os
from utils.constants import DATA_PATH
from utils.tools import get_contact_map
from utils_model.NetworkBlocks import ResnetBlockFC

##########################################################################################################################
## Latent Encoder
##########################################################################################################################

class LatentEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_size):
        super().__init__()
        self.block = ResnetBlockFC(size_in=in_dim, size_out=hidden_dim, size_h=hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_size)
        self.fc_logvar = nn.Linear(hidden_dim, latent_size)

        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        
    def forward(self, x):
        x = self.block(x, final_nl=True)
        return self.fc_mu(x), torch.clamp(self.fc_logvar(x), min=-10, max=9)

###########################################################################################################################
## DGCNN Encoder (from DRO-Grasp: https://github.com/zhenyuwei2003/DRO-Grasp)
###########################################################################################################################

def knn(x, k):
    x = x.permute(0, 2, 1)
    pairwise_distance = torch.cdist(x, x, p=2)      # (B, N, N)
    return pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, K)

def get_graph_feature(x, k=32):
    idx = knn(x, k=k)                                                               # (B, N, K)
    # batch_size, num_points, _ = idx.size()
    B, num_dims, N = x.size()

    idx_base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)

    x = x.permute(0, 2, 1)                                                          # (B, N, num_dims)
    x_flat = x.reshape(B * N, num_dims)                                             # (B * N, num_dims)
    feature = x_flat[idx, :].view(B, N, k, num_dims)                                # (B, N, K, num_dims)
    
    x = x.view(B, N, 1, num_dims).repeat(1, 1, k, 1)                                # (B, N, K, num_dims)
    feature = torch.cat((feature, x), dim=-1)                                       # (B, N, K, 2*num_dims)

    return feature.permute(0, 3, 1, 2).contiguous()                                 # (B, 2*num_dims, N, Ks)


class DGCNNEncoder(nn.Module):
    """
    Implementation extracted from DRO-Grasp (https://github.com/zhenyuwei2003/DRO-Grasp)
    """
    def __init__(self,
                 out_feat_dim=512,
                 layers_size=[12, 64, 64, 128, 256, 512],
                 knn=16):
        super(DGCNNEncoder, self).__init__()

        self.knn = knn
        self.layers_size = layers_size
        
        layers = []
        for i in range(len(layers_size) - 1):
            layers.append(nn.Conv2d(layers_size[i], layers_size[i+1], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(layers_size[i+1]))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.feature_extractor = nn.Sequential(*layers)

        layers_in_final_conv = int(sum(layers_size[1:]) + layers_size[-1])
        self.final_layer = nn.Sequential(
            nn.Conv1d(layers_in_final_conv, out_feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_feat_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        """
        :param in: (B, N, C)
        :return: (B, N, out_feat_dim)
        """
        x = x.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)
        B, _, N = x.size()

        x = get_graph_feature(x, k=self.knn)  # (B, 2*C, N, K)
        
        features: list[torch.Tensor] = []
        for i in range(0, len(self.feature_extractor), 3):
            x = self.feature_extractor[i](x)                          # (B, layer_size[i+1], N, K)
            x = self.feature_extractor[i + 1](x)
            x = self.feature_extractor[i + 2](x)
            features.append(x.max(dim=-1, keepdim=False)[0])    # (B, layer_size[i+1], N)

        global_feat = features[-1].mean(dim=-1, keepdim=True).repeat(1, 1, N).contiguous()  # (B, layer_size[-1], 1) -> (B, layer_size[-1], N)
        
        x = torch.cat(features + [global_feat], dim=1)          # (B, ..., N)
        x = self.final_layer(x)
        x = x.view(B, -1, N)                                                    # (B, layer_size[-1], N)

        return x.permute(0, 2, 1)  # (B, layer_size[-1], N) -> (B, N, layer_size[-1])

###########################################################################################################################
## Canonical Feature-Based Workspace Projection (based on BPS: Efficient Learning on Point Clouds With Basis Point Sets, ICCV 2019)
###########################################################################################################################
from utils_data.custom_bps import bps_torch, compute_aligned_dist_v2


class BPSFeatures(nn.Module):
    def __init__(self, robot_name, fixed_rt=True, bps_pts=8192):
        super(BPSFeatures, self).__init__()
        if fixed_rt:
            self.basis_set = torch.load(os.path.join(DATA_PATH, f'workspaces/{robot_name}_workspace_{bps_pts}_pts.pt'), weights_only=True)       # (M, 3)
        else:
            raise NotImplementedError("Only fixed_rt=True is implemented for now.")
        self.basis_shape = self.basis_set.shape[0]
        self.bps = bps_torch(custom_basis=self.basis_set, n_dims=3)
    
    def forward(self, x=None, y=None):
        """
        :param x: (B, N)
        :param y: (B, N, 6+C)
        :return: contact_map (B, M), label_map (B, M), bps_features (B, M, C)
        """
        B, N, C = y.shape
        C = C - 6

        self.basis_set = self.basis_set.to(y.device)

        if x is not None:
            contact_map = torch.zeros((B, self.basis_shape), device=y.device)  # (B, M, 1)
            label_map = torch.zeros((B, self.basis_shape), device=y.device)  # (B, M, 1)
            for b in range(B):
                mask = (x[b] > 0)
                if not mask.any():
                    continue
                target_points = y[b, mask, :6]
                target_labels = x[b, mask].unsqueeze(-1)    # (num_target, 1)
                cmap, _, lmap = get_contact_map(self.basis_set, target_points, target_labels, gamma=2.0, delta=0.1, contact_threshold=0.8)            # (1, M)
                contact_map[b] = cmap.squeeze(0)
                label_map[b] = lmap.squeeze(0)
        else:
            contact_map = None
            label_map = None

        if C > 0:
            # weighted average of k nearest neighbors' features
            k = min(5, N)   # k=5
            
            # dists = torch.cdist(self.basis_set.unsqueeze(0).expand(B, -1, -1), y[:, :, :3], p=2) # (B, M, N)
            dists = compute_aligned_dist_v2(X=y[:, :, :6], Y=self.basis_set.unsqueeze(0).expand(B, -1, -1), return_distances=True)  # (B, N, M)
            dists = dists.permute(0, 2, 1)  # (B, M, N)

            knn_dists, knn_idx = torch.topk(dists, k=k, dim=2, largest=False, sorted=False) # (B,M,k), (B,M,k)

            batch_idx_expand = torch.arange(B, device=y.device).view(-1, 1, 1).expand(-1, self.basis_shape, k)  # (B, M, k)
            feats = y[:, :, 6:]  # (B,N,C)
            neigh_feats = feats[batch_idx_expand, knn_idx]  # (B,M,k,C)

            # compute normalized weights and weighted average
            weights = torch.exp(-knn_dists)  # (B,M,k)
            # weights = weights / (torch.sum(weights, dim=2, keepdim=True) + 1e-8)
            weights = weights / k
            bps_features = torch.sum(weights.unsqueeze(-1) * neigh_feats, dim=2)  # (B,M,C)
        else:
            bps_features = None
        
        return contact_map, label_map, bps_features        # (B, M), (B, M, C)

