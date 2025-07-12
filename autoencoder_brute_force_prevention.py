import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
import time
import joblib
from datetime import datetime


# === Load Scaler and Model ===
scaler = joblib.load("scaler2.pkl")

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
model.load_state_dict(torch.load("autoencoder_model.pth"))
model.eval()

# === Telegram Bot Setup ==
import requests
def send_telegram_alert(ip):
    message = f"🚨 Bruteforce Attack Alert!: The malicious IP {ip} has been blocked!!! Please Blue Team investigate the incident!!!Send the blocked individual this verification link: http://192.168.100.170:5000/ "
    url = f"https://api.telegram.org/bot8100362782:AAHjQf2jC0yGjOtkOP-Rzs-ipb7YdP-A7mk/sendMessage"
    data = {"chat_id":7188548099, "text": message}
    
    try:
        response = requests.post(url, json=data)
        print(f"📨 Telegram alert sent for banned IP: {ip}")
        print(f"🔍 Response Code: {response.status_code}")
        print(f"🔍 Response Text: {response.text}")
    except Exception as e:
        print(f"❌ Failed to send Telegram alert: {e}")

# === Log Parser ===
def parse_logs():
    log_path = "/var/log/auth.log"
    entries = []

    try:
        with open(log_path, "r") as f:
            for line in f:
                match_failed = re.search(
                    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2}) .*? sshd(?:-session)?\[\d+\]: Failed password for (\w+) from (\d+\.\d+\.\d+\.\d+)",
                        line

                )
                match_accepted = re.search(
                    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+\d{2}:\d{2}) .*?sshd(?:-session)?\[\d+\]: Accepted password for (\w+) from (\d+\.\d+\.\d+\.\d+)", line
                )

                if match_failed:
                    timestamp_str,user, ip = match_failed.groups()
                    try:
                       timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                    except ValueError:
                       timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")
                    entries.append({"timestamp": timestamp.replace(year=datetime.now().year), "ip": ip, "status": 0})

                elif match_accepted:
                    timestamp_str, user, ip = match_accepted.groups()
                    try:
                       timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                    except ValueError:
                       timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")

                    entries.append({"timestamp": timestamp.replace(year=datetime.now().year), "ip": ip, "status": 1})

    except FileNotFoundError:
        print("Log file not found.")

    return pd.DataFrame(entries)

# === Sequence Builder ===
def create_sequences(df, seq_length=5):
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    df["failed_attempts"] = df["status"].apply(lambda x: 0 if x == 1 else 1)
    df[["time_diff", "failed_attempts"]] = scaler.transform(df[["time_diff", "failed_attempts"]])
    data = df[["time_diff", "failed_attempts"]].values

    sequences = []
    ip_refs = []

    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
        ip_refs.append(df["ip"].iloc[i + seq_length - 1])

  
    sequences = np.array(sequences, dtype=np.float32)
    sequences = torch.from_numpy(sequences)

    return sequences, ip_refs

# === Real-Time Detection Loop ===
THRESHOLD = 0.02  # You may tune this based on your training loss

while True:
    df_logs = parse_logs()

    if df_logs.empty or len(df_logs) < 5:
        print("⏳ Waiting...")
        time.sleep(5)
        continue

    df_logs = df_logs.sort_values(by="timestamp").reset_index(drop=True)
    print("✅ Parsed log entries:", len(df_logs))

    sequences, ip_refs = create_sequences(df_logs)
    print("✅ Sequences generated:", len(sequences))

    if len(sequences) == 0:
        print("⚠️ No valid sequences. Skipping...")
        time.sleep(5)
        continue

    with torch.no_grad():
        reconstructions = model(sequences)
        loss_fn = nn.MSELoss(reduction='none')
        losses = loss_fn(reconstructions, sequences).mean(dim=(1, 2)).numpy()
        
    for i, loss in enumerate(losses):
        ip = ip_refs[i]
        print(f"🔍 IP: {ip}, MSE Loss: {loss:.6f}")  # DEBUG LINE
        if loss < THRESHOLD:
            print(f"🚨 Detected brute-force from {ip} (MSE Loss: {loss:.4f})")
            os.system("sudo systemctl start fail2ban")
            os.system(f"sudo fail2ban-client set sshd banip {ip}")
            send_telegram_alert(ip)
          
                              
    print("✅ Monitoring, waiting for new logs...\n")
    time.sleep(5)
