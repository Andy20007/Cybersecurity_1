import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# Parameters
num_entries = 100000  # All normal
ip_pool = [f"192.168.100.{i}" for i in range(2, 254)]  # Shared IPs

# Start time
base_time = datetime.now()
entries = []

# === Only Normal Entries (status = 1) ===
for _ in range(num_entries):
    ip = random.choice(ip_pool)
    time_diff = np.random.uniform(400, 7500)  # Infrequent normal activity
    failed_attempts = np.random.randint(0, 3)  # Very few failures (0, 1, or 2)
    status = 1  # Successful login

    base_time += timedelta(seconds=time_diff)
    entries.append({
        "ip": ip,
        "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_diff": time_diff,
        "failed_attempts": failed_attempts,
        "status": status
    })

# === Convert to DataFrame and Save ===
df = pd.DataFrame(entries)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.to_csv("ai_dataset.csv", index=False)
print("Dataset with only normal login behavior saved as 'ai_dataset.csv'")

