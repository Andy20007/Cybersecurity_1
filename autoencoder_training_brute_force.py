import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# === Load Normal Data (status=1 only) ===
df = pd.read_csv("ai_dataset.csv", parse_dates=["timestamp"])
df = df[df["status"] == 1].sort_values(by="timestamp").reset_index(drop=True)

# === Feature Engineering ===
df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
features = ["time_diff", "failed_attempts"]

# === Normalize ===
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# === Create Sequences ===
def create_sequences(data, seq_len=5):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.stack(sequences)

sequence_data = create_sequences(df[features].values, seq_len=5)

# === Prepare PyTorch Dataloader ===
X_train, X_val = train_test_split(sequence_data, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=64)

# === Define LSTM Autoencoder ===
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, latent_size=16):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.latent = nn.Linear(hidden_size, latent_size)
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        encoded_seq, _ = self.encoder(x)
        latent = self.latent(encoded_seq[:, -1])
        hidden = self.decoder_input(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(hidden)
        return decoded

model = LSTMAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# === Training Loop ===
num_epochs = 60
import matplotlib.pyplot as plt

# Lists to store losses
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0]
            output = model(x)
            val_loss += criterion(output, x).item()

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
# === Save Model and Scaler ===
torch.save(model.state_dict(), "autoencoder_model.pth")
import joblib
joblib.dump(scaler, "scaler2.pkl")
