import torch
from torch import autograd

from gan import GanTrainer


class WGanGPTrainer(GanTrainer):
    def __init__(self, training_config, G, D, device):
        super().__init__(training_config, G, D, device)

    def update_discriminator(self, fake_data, fake_label, real_data, real_label):
        self.optimizer_D.zero_grad()
        fake_data_predict = self.D(fake_data)
        real_data_predict = self.D(real_data)
        # compute gradient penalty
        alpha = torch.rand(fake_data.shape[0], 1, 1, 1, device=self.device)
        interpolate_data = alpha * real_data + (1 - alpha) * fake_data
        interpolate_data.requires_grad_(True)  # or there will be an error
        gradient = autograd.grad(
            outputs=self.D(interpolate_data),
            inputs=interpolate_data,
            grad_outputs=torch.ones_like(real_data_predict),
            retain_graph=True,
            create_graph=True  # allowing to compute higher order derivative products
        )[0]  # the result is (gradient, )
        gradient_norm = gradient.view(gradient.shape[0], -1).norm(p=2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2) * self.config.lambda_gradient_penalty
        # compute loss
        loss_D = -torch.mean(real_data_predict) + torch.mean(fake_data_predict) + gradient_penalty
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D.detach()

    def update_generator(self, fake_data, label):
        self.optimizer_G.zero_grad()
        fake_data_predict = self.D(fake_data)

        loss_G = -torch.mean(fake_data_predict)
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.detach()
