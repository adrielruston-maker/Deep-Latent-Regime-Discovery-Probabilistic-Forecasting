import torch
import torch.nn as nn
import torch.nn.functional as F

"""
GRU encodes h -> latent variable -> probabilistic outcome (mu, sigma)
keep latent size small so analysis is easier...

"""
class GRULatentForecaster(nn.Module):
    def __init__(
            self,
            input_size=10,
            hidden_size=32,
            latent_size=3
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        # force z to be in R3
        self.latent = nn.Linear(
            hidden_size,
            latent_size
        )
        #probabilistic outcome wanted, Yt | zt ~ N(mu,sigma^2)
        #hence mu epsilon R but sigma > 0

        self.mu = nn.Linear(
            latent_size, 1
        )
        self.sigma = nn.Linear(
            latent_size, 1
        )
    # forward method
    def forward(self, x):
        gru_out, h_n = self.gru(x)

        h_final = h_n[-1]
        z = self.latent(h_final)
        #since flow is h -> z -> probabilistic outcome, it is okay to not tanh z here for the divergence problem
        mu = self.mu(z)
        raw_sigma = self.sigma(z)
        #need sigma > 0, choose an AF that gives a positive transformation
        sigma = F.softplus(raw_sigma) + 1e-6

        mu = mu.squeeze(-1)
        sigma = sigma.squeeze(-1)

        return mu, sigma, z

if __name__ == "__main__":
    model = GRULatentForecaster(
        input_size=10,
        hidden_size=32,
        latent_size=3
    )
    x = torch.rand(64, 64, 10)

    mu, sigma, z = model(x)

    print("mu:", mu.shape)
    print("sigma:", sigma.shape)
    print("z:", z.shape)

    print("\nSigma min", sigma.min().item())

    print("mu req grad", mu.requires_grad)
    print("sigma req grad", sigma.requires_grad)
    print("z req grad", z.requires_grad)
