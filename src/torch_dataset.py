import torch
from torch.utils.data import Dataset, DataLoader
from dataset import prepare_datasets

class MarketSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )
        #use common practice for DL -> float32
        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        )
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(
        batch_size=64,
        lookback=60,
        horizon=5,
):
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_dates, val_dates,
        test_dates, scaler
    ) = prepare_datasets(
        lookback=lookback,
        horizon=horizon,
    )

    train_dataset = MarketSequenceDataset(X_train, y_train)

    val_dataset = MarketSequenceDataset(X_val, y_val)

    test_dataset = MarketSequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dates,
        val_dates,
        test_dates,
        scaler
    )

if __name__ == "__main__":
    (
        train_loader, val_loader,
        test_loader, train_dates,
        val_dates, test_dates, scaler
    ) = create_dataloaders()

    X_batch, y_batch = next(iter(train_loader))

    print("X Batch:", X_batch.shape)
    print("y Batch:", y_batch.shape)
#refactor done




