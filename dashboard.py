from flask import Flask, render_template
import re

app = Flask(__name__)

# Regex patterns
SUCCESS_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+\d{2}:\d{2}) .*?sshd(?:-session)?\[\d+\]: Accepted password for (\w+) from (\d+\.\d+\.\d+\.\d+)"
)

FAILED_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2}) .*? sshd(?:-session)?\[\d+\]: Failed password for (\w+) from (\d+\.\d+\.\d+\.\d+)"
)

def parse_auth_log():
    log_entries = []
    log_path = "/var/log/auth.log"

    try:
        with open(log_path, "r") as file:
            for line in file:
                match_success = SUCCESS_PATTERN.search(line)
                match_failed = FAILED_PATTERN.search(line)

                if match_success:
                    timestamp, user, ip = match_success.groups()
                    log_entries.append({
                        "timestamp": timestamp,
                        "user": user,
                        "ip": ip,
                        "status": "Accepted"
                    })
                elif match_failed:
                    timestamp, user, ip = match_failed.groups()
                    log_entries.append({
                        "timestamp": timestamp,
                        "user": user,
                        "ip": ip,
                        "status": "Failed"
                    })

    except Exception as e:
        print(f"Error reading log file: {e}")

    return log_entries

@app.route("/")
def dashboard():
    logs = parse_auth_log()
    return render_template("dashboard.html", logs=logs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
