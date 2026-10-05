import math
import torch
"""
recall project assumption Yt | Xt ~ N(mu, sigma^2)
Negative loss likelihood
recall p(yt | xt) ~ expected pdf
log(p(yt | xt)) = -log(sigma) -1/2*log(2pi) - (yt - mu)^2 / 2sigma^2
negate for NLL then take mean (1/N * SIGMA i=1 to N (NLL))
"""

def gaussian_nll(mu, sigma, target):
    nll = (
        torch.log(sigma) +
        ((target - mu) ** 2) / (2 * sigma**2) +
        0.5 * math.log(2 * math.pi)
    )
    return nll.mean()

if __name__ == "__main__":
    from models.gru_forecaster import GRULatentForecaster

    model = GRULatentForecaster(
        input_size=10,
        hidden_size=32,
        latent_size=3
    )

    X = torch.randn(64, 64, 10)
    y = torch.randn(64) * 0.02

    mu, sigma, z = model(X)

    loss = gaussian_nll(mu, sigma, y)

    print("Loss:", loss)
    print("Loss Shape", loss.shape)
    print("Loss req grad", loss.requires_grad)
    print("Loss is finite", torch.isfinite(loss).item())

    loss.backward()
    print("Gru grad exists", model.gru.weight_ih_l0.grad is not None)
    print("Latent grad exists", model.latent.weight.grad is not None)
    print("Mu grad exists", model.mu.weight.grad is not None)
    print("sigma grad exists", model.sigma.weight.grad is not None)

    # L -> (mu,sigma) -> z -> h -> GRU sanity check shows model is capable of learning