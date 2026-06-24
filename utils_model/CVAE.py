import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import time

from utils_model.Encoder import LatentEncoder, BPSFeatures, DGCNNEncoder
from utils_model.NetworkBlocks import MLP
from utils_model.Decoder import AdaLNDecoder
from utils_model.TransformerFlashAttn import Transformer, PoolingAttention
from utils_model.LossCVAE import LossCVAELabels


class CVAE(L.LightningModule):
    def __init__(self, cfg):
        super(CVAE, self).__init__()

        self.cfg = cfg
        self.latent_size = cfg['latent_size']

        
        self.pointwise_feat = DGCNNEncoder(
            out_feat_dim=self.cfg['dgcnn_features_size'],
            layers_size=self.cfg['dgcnn_layers_size'],
            knn=self.cfg['dgcnn_knn']
        )
        self.type_embedder = nn.Embedding(self.cfg['num_types'], self.cfg['type_embedding_size'])

        self.bps_feat = BPSFeatures(robot_name=self.cfg['robot_name'], fixed_rt=self.cfg['fixed_rt'], bps_pts=self.cfg['bps_pts'])
        
        self.transformer = Transformer(
            in_channels=self.cfg['dgcnn_features_size'] + self.cfg['type_embedding_size'],
            n_blocks=self.cfg['transformer_n_blocks'],
            n_heads=self.cfg['transformer_n_heads'],
            dropout=self.cfg['transformer_dropout']
        )

        self.x_embedder = MLP(
            in_channels=self.cfg['num_zones'],
            hidden_channels=self.cfg['x_encoder_hidden_layers'] + [self.cfg['x_features_size']],
            dropout=0.0,
            bias=True,
        )

        self.pooling = PoolingAttention(
            dim_in=self.cfg['dgcnn_features_size'] + self.cfg['type_embedding_size'] + self.cfg['x_features_size'],
            dim_out=self.cfg['pooling_size'], 
            num_heads=self.cfg['pooling_n_heads'], 
            num_seeds=1
        )

        self.latent_encoder = LatentEncoder(
            in_dim=self.cfg['pooling_size'],
            hidden_dim=self.cfg['latent_hidden_dim'],
            latent_size=self.cfg['latent_size']
        )

        self.loss_criterion = LossCVAELabels(
            beta_range=self.cfg['beta_range'],
            beta_cycle_start=self.cfg['beta_cycle_start'],
            beta_cycle_length=self.cfg['beta_cycle_length'],
            sigmoid_scale=self.cfg['sigmoid_scale'],
            is_cyclique=self.cfg['is_cyclique'],
            batchsize=self.cfg['batch_size'], 
            beta_fixed=self.cfg['beta_fixed']
        )  

        self.decoder = AdaLNDecoder(
            feature_dim=self.cfg['dgcnn_features_size'] + self.cfg['type_embedding_size'],
            latent_dim=self.cfg['latent_size'],
            num_classes=self.cfg['num_zones']
        )

        self.train_logs = {
            'loss': [],
            'recon': [],
            'kld': []
        }

    def pre_encoding(self, x=None, y=None):
        """
        :param x: labels (B, N)
        :param y: condition = PCD (B, N, 6)
        :return: contact_map (B, M), condition = bps_features (B, M, C)
        """
        y = torch.cat((y, self.pointwise_feat(y)), dim=-1)                         # (B, N, 6+features_size)
        _, x, y = self.bps_feat(x, y)                                                           # (B, M), (B, M, features_size)
        return x, y

    def forward(self, x, y, t):
        """
        :param x: pre-encoded x (B, M)
        :param y: pre-encoded condition (B, M, features_size)
        :param t: grasp type indices (B,)
        :return:
        """
        # Condition
        t = torch.clamp(t, 0, self.cfg['num_types'] - 1)
        t_emb = self.type_embedder(t).unsqueeze(1).repeat(1, y.size(1), 1)                  # (B, 1, features_size)
        y_t = self.transformer(torch.cat([y, t_emb], dim=-1))                               # (B, M, features_size)
        # Input labels embedding
        x_one_hot = F.one_hot(x.long(), num_classes=self.cfg['num_zones']).float()          # (B, M, num_zones)
        x_embed = self.x_embedder(x_one_hot)                                                # (B, M, x_features_size)
        # Latent
        fused_y_t_x = torch.cat([y_t, x_embed], dim=-1)                                     # (B, M, features_size + x_features_size)
        latent = self.pooling(fused_y_t_x)                                                  # (B, pooling_size)
        means, logvars = self.latent_encoder(latent)                                        # (B, latent_size), (B, latent_size)
        z_latent_code = torch.distributions.normal.Normal(means, torch.exp(0.5 * logvars)).rsample()
        # Decode
        x_hat = self.decoder(y_t, z_latent_code)                                              # (B, M)
        return x_hat, means, logvars, z_latent_code

    def inference(self, y, t, z_latent_code):
        """
        :param y: condition = PCD (B, N, 6)
        :param t: grasp type indices (B,)
        :param z_latent_code: (B, latent_size)
        :return x_hat: (B, M)
        """
        _, y = self.pre_encoding(y=y)                                                       # (B, M, features_size)
        t_emb = self.type_embedder(t).unsqueeze(1).repeat(1, y.size(1), 1)                  # (B, 1, features_size)
        y_t = self.transformer(torch.cat([y, t_emb], dim=-1))                               # (B, M, features_size)
        x_hat = self.decoder(y_t, z_latent_code)
        return x_hat

    def training_step(self, batch, batch_idx):
        y, _, x_gt, t, _ = batch
        x = x_gt.squeeze(-1)                                                                # (B, M)
        x, y = self.pre_encoding(x, y)                                                      # (B, M), (B, M, features_size)
        x_hat, means, logvars, _ = self(x, y, t)
        loss, recon, kld = self.loss_criterion(means, logvars, x.long(), x_hat)
        self.train_logs['loss'].append(loss.item())
        self.train_logs['recon'].append(recon.item())
        self.train_logs['kld'].append(kld.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.cfg['learning_rate']))
        return optimizer

    def on_train_epoch_end(self):
        self.loss_criterion.apply_iter()
        self.logger.experiment.add_scalar('beta', self.loss_criterion.beta, self.current_epoch)
        self.log('loss', sum(self.train_logs['loss']) / len(self.train_logs['loss']), logger=False, sync_dist=True)
        for key in self.train_logs.keys():
            avg_train_loss = sum(self.train_logs[key]) / len(self.train_logs[key])
            self.logger.experiment.add_scalars(key, {'train': avg_train_loss}, self.current_epoch)
            # Reset logs
            self.train_logs[key] = []
