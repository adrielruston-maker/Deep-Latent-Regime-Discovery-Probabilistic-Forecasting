import torch

from models.gru_forecaster import GRULatentForecaster
from loss import gaussian_nll
from torch_dataset import create_dataloaders
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / 'best_gru_model.pt'

def train_one_epoch(
        model,
        train_loader,
        optimizer,
        device
):
    model.train()

    total_loss = 0.0
    total_sigma = 0.0
    total_samples = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        #forward pass
        mu, sigma, z = model(X_batch)
        loss = gaussian_nll(mu, sigma, y_batch)
        #backprop
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #optimizer
        optimizer.step()


        batch_size = X_batch.size(0)

        total_loss += loss.item() * batch_size
        total_sigma += sigma.detach().sum().item() #detach() so pytorch doesnt retain
        total_samples += batch_size
    avg_loss = total_loss / total_samples
    avg_sigma = total_sigma / total_samples
    return avg_loss, avg_sigma

#recall no optimizer.zero_grad(), loss.backward()... val must not affect weights
def validate_one_epoch(
        model,
        val_loader,
        device
):
    model.eval()

    total_loss = 0.0
    total_sigma = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            mu, sigma, z = model(X_batch)

            loss = gaussian_nll(mu, sigma, y_batch)

            batch_size = X_batch.size(0)

            total_loss += loss.item() * batch_size
            total_sigma += sigma.detach().sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_sigma = total_sigma / total_samples

        return avg_loss, avg_sigma

def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        checkpoint_path,
        num_epochs=50
):

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sigma": [],
        "val_sigma": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_sgima = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )
        val_loss, val_sigma = validate_one_epoch(
            model,
            val_loader,
            device
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                checkpoint_path
            )
        #theta_e -> theta_e+1 -> L_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_sigma"].append(train_sgima)
        history["val_sigma"].append(val_sigma)
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Train Sigma: {train_sgima:.6f} | "
            f"Val Sigma: {val_sigma:.6f}"
            f"\nBest Validation Loss: {best_val_loss:.6f} at epoch {best_epoch} "
        )
    return history
#check for argmin L_val later
#plot
def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train NLL"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation NLL"
    )

    best_epoch = history["best_epoch"]
    best_val_loss = history["best_val_loss"]

    plt.scatter(
        best_epoch,
        best_val_loss,
        label=f"Best validation epoch {best_epoch}",
        zorder=5
    )

    plt.xlabel("Epoch")
    plt.ylabel("Gaussian NLL")
    plt.title("Training and Validation NLL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_sigma_history(history):
    epochs = range(1, len(history["train_sigma"]) + 1)

    plt.plot(
        epochs,
        history["train_sigma"],
        label="Train Mean Sigma"
    )
    plt.plot(
        epochs,
        history["val_sigma"],
        label="Validation Mean Sigma"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Predicted Sigma")
    plt.title("Predicted Uncertainty During Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cude" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device: ", device)

    (
        train_loader, val_loader,
        test_loader, train_dates,
        val_dates, test_dates, scaler
    ) = create_dataloaders(
        batch_size=64,
        lookback=60,
        horizon=5
    )

    model = GRULatentForecaster(
        input_size=10,
        hidden_size=32,
        latent_size=3,
    ).to(device)

    #AdamW since GRU
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_path=CHECKPOINT_PATH,
        num_epochs=50
    )
    plot_training_history(history)
    plot_sigma_history(history)
#after first run, mean sigma changed far less compared to NLL
#refactor the 5 day target eventually




