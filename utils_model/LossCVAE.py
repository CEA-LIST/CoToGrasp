import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossCVAELabels(nn.Module):
    def __init__(self, batchsize, beta_cycle_start, beta_cycle_length,
                beta_range=[0.0001, 0.1], sigmoid_scale=20.0, is_cyclique=False, beta_fixed=None):
        super(LossCVAELabels, self).__init__()
        
        self.beta_range = [float(beta_range[0]), float(beta_range[1])]
        self.beta = self.beta_range[0]
        self.beta_cycle_start = beta_cycle_start
        self.cycle_length = beta_cycle_length
        self.sigmoid_scale = sigmoid_scale
        self.is_cyclique = is_cyclique
        self.beta_fixed = beta_fixed

        self.batchsize = batchsize

        self.iter_counter = 0

    def forward(self, means, logvars, x_gt, x_hat):
        """
        :param means:
        :param logvars:
        :param x_gt: B x N x C
        :param x_hat: B x N x C
        :return:
        """
        # Prevent potential issues with float16
        means = means.to(torch.float32)
        logvars = logvars.to(torch.float32)
        x_hat = x_hat.to(torch.float32)

        # Loss KLD
        logvars = torch.clamp(logvars, min=-10, max=10)
        loss_kld = -0.5 *  torch.mean(1 + logvars - means.pow(2) - torch.exp(logvars))

        # Recon Loss
        x_hat = x_hat.reshape(-1, x_hat.shape[-1])      # (B*N, C)
        x_gt = x_gt.reshape(-1).long()                  # (B*N)
        loss_recon = F.cross_entropy(x_hat, x_gt)

        if torch.isnan(loss_recon) or torch.isnan(loss_kld):
            # If we hit a NaN here, it means the model weights are already corrupted
            print("Loss recon or KLD is NaN.")
            return torch.tensor(0.0, device=means.device, requires_grad=True), loss_recon, loss_kld

        # Loss
        loss = self.beta * loss_kld + loss_recon
        return loss, loss_recon, loss_kld

    def apply_iter(self):
        if self.beta_fixed:
            self.beta = self.beta_fixed
        else:
            if self.iter_counter < self.beta_cycle_start:
                self.beta = self.beta_range[0]
            else:
                if self.is_cyclique:
                    phase = ((self.iter_counter - self.beta_cycle_start) % self.cycle_length) / self.cycle_length
                    self.beta = self.beta_range[0] + (self.beta_range[1] - self.beta_range[0]) / (1 + math.exp(-self.sigmoid_scale * (phase - 0.5)))    # Apply sigmoid function to smoothly transition in the beta_range
                else:
                    if self.iter_counter < self.cycle_length + self.beta_cycle_start:
                        phase = ((self.iter_counter - self.beta_cycle_start) % self.cycle_length) / self.cycle_length
                        self.beta = self.beta_range[0] + (self.beta_range[1] - self.beta_range[0]) / (1 + math.exp(-self.sigmoid_scale * (phase - 0.5)))
                    else:
                        self.beta = self.beta_range[1]
        self.iter_counter += 1

