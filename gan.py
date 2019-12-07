import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader

from utils import ImageDataset, init_logger


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1.0)
        m.bias.data.fill_(0)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        pass


class Generator(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.img_size = tuple(model_config["img_size"])
        self.latent_size = tuple(model_config["latent_size"])
        self.model = nn.Sequential(
            nn.Linear(self.latent_size[0], 1200),
            nn.ReLU(True),
            nn.Linear(1200, 1200),
            nn.ReLU(True),
            nn.Linear(1200, np.prod(self.img_size)),
            nn.Sigmoid()  # [0, 1]
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, *self.img_size)
        return x

    def get_input_shape(self):
        return self.latent_size

    def init_weights(self):
        pass
        #self.apply(weights_init)


class Maxout(nn.Module):
    def __init__(self, k, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(k)])

    def forward(self, x):
        maxout = self.linears[0](x)
        for _, layer in enumerate(self.linears, start=1):
            maxout = torch.max(maxout, layer(x))
        return maxout


class Discriminator(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.img_size = tuple(model_config["img_size"])
        self.last_layer_sigmoid = model_config["D_last_layer_sigmoid"]

        self.maxout1 = Maxout(5, np.prod(self.img_size), 240)
        self.maxout2 = Maxout(5, 240, 240)
        self.fc = nn.Linear(240, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.maxout1(x)
        x = F.dropout(x, training=self.training)
        x = self.maxout2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        if self.last_layer_sigmoid:
            x = torch.sigmoid(x)
        return x

    def init_weights(self):
        pass
        #self.apply(weights_init)


class GanTrainer(object):
    def __init__(self, training_config, G, D, device):
        self.device = device
        self.config = SimpleNamespace(**training_config)
        self.exp_dir = Path("2_experiments") / self.config.exp_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.stat = {}  # training statistics
        self.logger = init_logger("training", self.exp_dir / "log")  # training logger

        dataset = ImageDataset(self.config.dataset)
        self.data_loader = DataLoader(dataset, batch_size=self.config.batch_size,
                                 shuffle=True, num_workers=self.config.num_loader_worker)

        self.G = G.to(self.device)
        self.D = D.to(self.device)
        self.logger.info("Latent Size: {}".format(self.G.get_input_shape()))
        self.optimizer_G = self.get_optimizer(self.G, self.config.optimizer_G)
        self.optimizer_D = self.get_optimizer(self.D, self.config.optimizer_D)

        self.loss = nn.BCELoss().to(self.device)

    def get_optimizer(self, net, optimizer_config):
        config = SimpleNamespace(**optimizer_config)

        if config.optimizer == "Adam":
            self.logger.info(
                "Optimizer: Adam    " +
                "LR: {}    ".format(config.lr) +
                "betas: {}".format(tuple(config.beta))
            )
            return Adam(net.parameters(), lr=config.lr, betas=tuple(config.beta))
        elif config.optimizer == "RMSprop":
            self.logger.info("Optimizer: Adam    " + "LR: {}".format(config.lr))
            return RMSprop(net.parameters(), lr=config.lr)
        else:
            print("No optimizer named {}".format(config.optimizer))
            exit(-1)

    def save_checkpoint(self, checkpoint_path, epoch):
        self.logger.info("Saving checkpoint: " + str(checkpoint_path))
        checkpoint = {
            "epoch": epoch,  # completed epoch
            "stat": self.stat,
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        self.logger.info("Loading checkpoint: " + str(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.stat = checkpoint["stat"]
        self.G.load_state_dict(checkpoint["G"])
        self.D.load_state_dict(checkpoint["D"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        return checkpoint["epoch"]

    def find_last_checkpoint(self):  # return None if no checkpoint found
        last_epoch = -1
        for checkpoint in self.exp_dir.glob("ckp_*.pth"):
            epoch = int(checkpoint.stem.split("_")[1])
            last_epoch = max(last_epoch, epoch)
        return self.exp_dir / "ckp_{:0>4d}.pth".format(last_epoch) if last_epoch != -1 else None

    def sample_from_generator(self, sample_path):  # generate some samples from G and save to sample_file
        self.logger.info("Sample from Generator and save to " + str(sample_path))
        self.G.eval()  # switch to eval mode
        z = torch.randn((self.config.batch_size, *self.G.get_input_shape()), device=self.device)
        output = self.G(z).detach().cpu().numpy()
        np.save(str(sample_path), output)
        self.G.train()  # switch back to training mode

    def train(self):
        # set training attribute as True
        self.G.train()
        self.D.train()
        # init weights
        self.G.init_weights()
        self.D.init_weights()
        # init stat
        self.stat["training_loss_G"] = []
        self.stat["training_loss_D"] = []
        # load checkpoint if exists
        start_epoch = 0
        checkpoint_file = self.find_last_checkpoint()
        if checkpoint_file is not None:
            start_epoch = self.load_checkpoint(checkpoint_file) + 1
        # start training
        self.logger.info("Training on device: " + str(self.device))
        self.logger.info("Total #epoch: " + str(self.config.num_epoch) + "    Start epoch: " + str(start_epoch))
        for epoch in range(start_epoch, self.config.num_epoch):
            start_time = time.time()
            self.train_one_epoch()
            finish_time = time.time()
            self.logger.info(
                "Epoch {}    ".format(epoch) +
                "Loss G: {:f}    ".format(self.stat["training_loss_G"][-1]) +
                "Loss D: {:f}    ".format(self.stat["training_loss_D"][-1]) +
                "Elapsed time: {:.2f}".format(finish_time - start_time)
            )
            # save checkpoint
            if epoch % self.config.save_every_n_epoch == 0 or epoch == self.config.num_epoch - 1:
                self.save_checkpoint(self.exp_dir / "ckp_{:0>4d}.pth".format(epoch), epoch)
                self.sample_from_generator(self.exp_dir / "sample_{:0>4d}.npy".format(epoch))
        # save stat after training
        torch.save(self.stat, self.exp_dir / "stat.pth")

    def train_one_epoch(self):
        training_loss_G, training_loss_D = [], []
        for i, training_data_batch in enumerate(self.data_loader):
            batch_size = training_data_batch.shape[0]
            # Training Discriminator
            # Sample minibatch of m noise samples from noise prior p_g(z)
            z = torch.randn((batch_size, *self.G.get_input_shape()), device=self.device)  # Standard Normal
            fake_data = self.G(z)
            fake_label = torch.zeros((batch_size, 1), device=self.device)
            # Sample minibatch of m examples from data generating distribution p_data(x)
            real_data = training_data_batch.to(self.device)
            real_label = torch.ones((batch_size, 1), device=self.device)
            # Update the Discriminator
            # use detach() to avoid gradient coming into Generator
            training_loss = self.update_discriminator(fake_data.detach(), fake_label, real_data, real_label)
            training_loss_D.append(training_loss.cpu().numpy())

            # Train Discriminator several times, and then train Generator one time
            if i % self.config.discriminator_training_steps_per_iter != 0:
                continue

            # Training Generator
            # Sample minibatch of m noise samples from noise prior p_g(z)
            z = torch.randn((batch_size, *self.G.get_input_shape()), device=self.device)  # Standard Normal
            fake_data = self.G(z)
            label = torch.ones((batch_size, 1), device=self.device)
            # Update the Generator
            training_loss = self.update_generator(fake_data, label)
            training_loss_G.append(training_loss.cpu().numpy())

        # record training loss in every epoch
        self.stat["training_loss_G"].append(np.mean(training_loss_G))
        self.stat["training_loss_D"].append(np.mean(training_loss_D))

    def update_discriminator(self, fake_data, fake_label, real_data, real_label):
        self.optimizer_D.zero_grad()
        fake_data_predict = self.D(fake_data)
        real_data_predict = self.D(real_data)
        loss_D = (self.loss(fake_data_predict, fake_label) + self.loss(real_data_predict, real_label)) / 2
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D.detach()

    def update_generator(self, fake_data, label):
        self.optimizer_G.zero_grad()
        fake_data_predict = self.D(fake_data)
        # an alternative loss -log(D) is used here
        loss_G = self.loss(fake_data_predict, label)
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.detach()
