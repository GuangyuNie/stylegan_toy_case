# import
import os, argparse, datetime, math
from random import randint
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


class toy:
    def __init__(self):
        self.num_samples = opt.sample
        self.batch_size = opt.bs
        self.device = torch.device("cuda:0")


    def data_gen(self):
        """generate training data"""
        data_np = np.empty((self.num_samples,3))
        r = 100
        p = 10
        for i in range(self.num_samples):
            t = 3*np.pi*np.random.rand(1)
            x = r*np.cos(t)
            y = r*np.sin(t)
            z = p*t+np.random.uniform(low=-2,high=2)
            data_np[i] = [x,y,z]
        data = torch.from_numpy(data_np).to(toy.device)
        return data


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x.float())
        return output

class EqualizedLinear(nn.Module):
    """
    This applies the equalized weight method for a linear layer
    input:
      input features
      output features
      bias
    return:
      y = wx+b
    """

    def __init__(self, in_features, out_features, bias=0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight(), bias=self.bias)

class EqualizedWeight(nn.Module):
    """
    Introduced in https://arxiv.org/pdf/1710.10196.pdf
    Instead of draw weight from a normal distribution of (0,c),
    It draws from (0,1) instead, and times the constant c.
    so that when using optimizer like Adam which will normalize through
    all gradients, which may be a problem if the dynamic range of the weight
    is too large.
    input:
    shape: [in_features,out_features]
    return:
    Randomized weight for corresponding layer
    """

    def __init__(self, shape):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class Generator(nn.Module):
    def __init__(self, features, n_layers):
        super().__init__()

        # Mapping Network
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

        # Synthesis Network: input 2D data, output 3D data
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
        )

    # Get the w = f(z)
    def get_latent(self,x):
        return self.net(x.float())

    # input w instead of input z
    def input_latent(self,x):
        return self.model(x)

    # Get model output
    def forward(self, x):
        x = self.net(x.float())
        output = self.model(x)
        return output


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epoch",
        type=int,
        default=10000,
        help="number of training epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
        default=0.0001
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="training_sample",
        default=10000
    )
    parser.add_argument(
        "--bs",
        type=int,
        help="batch_size",
        default=128
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="logdir",
        default="./test_result/"
    )
    parser.add_argument(
        "--lm",
        type=float,
        help="lambda for KL divergence term",
        default=0.1
    )
    parser.add_argument(
        "--lam",
        type=float,
        help="lambda for KL divergence term",
        default=0.0001
    )
    return parser


if __name__ == '__main__':
    # Initializing
    parser = get_parser()
    opt, _ = parser.parse_known_args()
    toy = toy()
    data = toy.data_gen()
    batch_size = toy.batch_size
    lr = opt.lr
    num_epochs = opt.epoch
    loss_function = nn.MSELoss()
    discriminator = Discriminator().to(toy.device)
    generator = Generator(features=2, n_layers=2).to(toy.device)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    # Create log dir
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # get time
    logdir = os.path.join(opt.logdir,now)
    os.makedirs(logdir)
    latent_path = os.path.join(logdir,'latent_space/')
    ckpt_path = os.path.join(logdir,'ckpt/')
    visual_path = os.path.join(logdir,'pixel_space/')
    os.makedirs(latent_path)
    os.makedirs(ckpt_path)
    os.makedirs(visual_path)

    # Load training data
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True,drop_last=True
    )

    # Training
    for epoch in tqdm(range(num_epochs)):
        for n, real_samples in enumerate(train_loader):
            # Data for training the discriminator
            real_samples_labels = torch.ones((batch_size, 1)).to(toy.device)
            z = torch.randn(batch_size,2).to(toy.device)
            generated_samples = generator(z).to(toy.device)
            generated_samples_labels = torch.zeros((batch_size, 1)).to(toy.device)
            all_samples = torch.cat((real_samples, generated_samples)).to(toy.device)
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            ).to(toy.device)

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples).to(toy.device)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Training the generator
            generator.zero_grad()
            z = torch.randn(batch_size,2).to(toy.device)
            latent = generator.get_latent(z)

            generated_samples = generator.input_latent(latent).to(toy.device)
            output_discriminator_generated = discriminator(generated_samples)

            """generator loss, add additional loss term here, KL-regularization is currently used to pump the ball"""
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels) + \
                             opt.lm * F.kl_div(latent, z)

            loss_generator.backward()
            optimizer_generator.step()


            # Show loss
            if epoch % 10 == 0 and n == 0:
                z = torch.randn(batch_size,2).to(toy.device)
                latent = generator.get_latent(z).detach().cpu().numpy()
                ax = plt.axes()
                ax.scatter(latent[:, 0], latent[:, 1])
                ax.figure.savefig(os.path.join(latent_path, 'latent_{}.png'.format(epoch)))
                ax.figure.clf()
                generated_samples = generated_samples.detach().cpu()
                real_samples = real_samples.detach().cpu()
                ax = plt.axes(projection='3d')
                ax.scatter(generated_samples[:, 0], generated_samples[:, 1], generated_samples[:, 2],label='estimated data')
                ax.scatter(real_samples[:, 0], real_samples[:, 1], real_samples[:, 2], label='real data')
                ax.figure.savefig(os.path.join(visual_path, 'pixel_{}.png'.format(epoch)))
                ax.legend(loc="upper right")
                ax.figure.clf()
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                torch.save(generator.state_dict(), os.path.join(ckpt_path, '{}.pt'.format(epoch)))

    generated_samples = generated_samples.detach().cpu()
    real_samples = real_samples.detach().cpu()
    data = data.detach().cpu()
    ax = plt.axes(projection='3d')
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], generated_samples[:, 2], )
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], )
    ax.scatter(real_samples[:, 0], real_samples[:, 1], real_samples[:, 2], )
    plt.show()