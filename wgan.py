import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gan import GanTrainer, Maxout


class WGanTrainer(GanTrainer):
    def __init__(self, training_config, G, D, device):
        super().__init__(training_config, G, D, device)

    def update_discriminator(self, fake_data, fake_label, real_data, real_label):
        self.optimizer_D.zero_grad()
        fake_data_predict = self.D(fake_data)
        real_data_predict = self.D(real_data)
        # add negative here because of gradient ascent
        loss_D = -(torch.mean(real_data_predict) - torch.mean(fake_data_predict))
        loss_D.backward()
        self.optimizer_D.step()
        # clipping
        for p in self.D.parameters():
            p.data.clamp_(-self.config.clip_limit, self.config.clip_limit)
        return loss_D.detach()

    def update_generator(self, fake_data, label):
        self.optimizer_G.zero_grad()
        fake_data_predict = self.D(fake_data)

        loss_G = -torch.mean(fake_data_predict)
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.detach()
