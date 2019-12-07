import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
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

        self.color_channel_dim = self.img_size[0]
        self.init_h, self.init_w = self.img_size[1] // 16, self.img_size[2] // 16
        self.gf_dim = 16  # Dimension of generator filters in first conv layer.

        self.fc = nn.Linear(self.latent_size[0], 8 * self.gf_dim * self.init_h * self.init_w)
        self.bn = nn.BatchNorm2d(8 * self.gf_dim)
        self.deconv_model = nn.Sequential(  # fractionally-strided convolutions
            # conv 1
            # size: output = (input - 1) * stride - 2 * padding + kernel size
            # kernel size = 4, stride = 2, padding = 1
            nn.ConvTranspose2d(8 * self.gf_dim, 4 * self.gf_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * self.gf_dim),
            nn.ReLU(True),
            # conv 2
            nn.ConvTranspose2d(4 * self.gf_dim, 2 * self.gf_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * self.gf_dim),
            nn.ReLU(True),
            # conv 3
            nn.ConvTranspose2d(2 * self.gf_dim, self.gf_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gf_dim),
            nn.ReLU(True),
            # conv 4
            nn.ConvTranspose2d(self.gf_dim, self.color_channel_dim, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        # project and reshape
        x = self.fc(x)
        x = x.view(-1, 8 * self.gf_dim, self.init_h, self.init_w)  # reshape
        x = self.bn(x)
        x = F.relu(x)

        x = self.deconv_model(x)
        return x

    def get_input_shape(self):
        return self.latent_size

    def init_weights(self):
        self.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.img_size = tuple(model_config["img_size"])
        self.last_layer_sigmoid = model_config["D_last_layer_sigmoid"]

        self.color_channel_dim = self.img_size[0]
        self.df_dim = 16  # Dimension of discriminator filters in first conv layer.

        self.conv_model = nn.Sequential(
            # conv 1
            nn.Conv2d(self.color_channel_dim, self.df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # conv 2
            nn.Conv2d(self.df_dim, 2 * self.df_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * self.df_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # conv 3
            nn.Conv2d(2 * self.df_dim, 4 * self.df_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * self.df_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # conv 4
            nn.Conv2d(4 * self.df_dim, 8 * self.df_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8 * self.df_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(8 * self.df_dim * self.img_size[1] // 16 * self.img_size[2] // 16, 1)

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        if self.last_layer_sigmoid:
            x = torch.sigmoid(x)
        return x

    def init_weights(self):
        self.apply(weights_init)
