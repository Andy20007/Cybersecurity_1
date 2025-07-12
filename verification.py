from  flask import Flask, request, render_template, redirect
import time
import os
import random
import requests

app = Flask(__name__)


otp_store = {} # Store OTPs : { ip: (otp, timestamp)}

# Telegram bot settings
BOT_TOKEN = "8030096468:AAGatK508MYmJhffI65eouadbK4A-DtEPb4"
CHAT_ID = "7188548099"

#OTP expiry time
OTP_EXPIRY_SECONDS = 120

def send_otp_telegram(bot_token, chat_id, ip, otp):
     BOT_TOKEN = "8030096468:AAGatK508MYmJhffI65eouadbK4A-DtEPb4"
     CHAT_ID = "7188548099"

     message = f"Your OTP to unblock IP {ip} is {otp}. It expires in 2 minutes." 
     url = f"https://api.telegram.org/bot8030096468:AAGatK508MYmJhffI65eouadbK4A-DtEPb4/sendMessage"
     requests.post(url, data={"chat_id": chat_id, "text": message})
 
@app.route("/", methods=["GET", "POST"])
def verify():
     BOT_TOKEN = "8030096468:AAGatK508MYmJhffI65eouadbK4A-DtEPb4"
     CHAT_ID = "7188548099"

     if request.method == "POST":
           ip = request.form.get("ip")
  
        # Generate OTP
           otp = str(random.randint(1000000, 9999999))
           timestamp = time.time()
           otp_store[ip] = (otp, time.time())

        # Send via Telegram
           send_otp_telegram(BOT_TOKEN, CHAT_ID, ip, otp)

           return redirect(f"/verify-otp?ip={ip}")
     else:

         return render_template("verify.html") 


@app.route("/verify-otp", methods=["GET", "POST"])
def otp():
    ip = request.args.get("ip")
    BOT_TOKEN = "8030096468:AAGatK508MYmJhffI65eouadbK4A-DtEPb4"
    CHAT_ID = "7188548099"
    

    if request.method == "POST":
        otp_entered = request.form["otp"]
        data = otp_store.get(ip)

        if not data:
            return "No OTP found for this IP."

        correct_otp, timestamp = data

        # Check expiry
        if time.time() - timestamp > OTP_EXPIRY_SECONDS:
            # Auto-generate new OTP
            new_otp = str(random.randint(1000000, 9999999))
            otp_store[ip] = (new_otp, time.time())
            send_otp_telegram(BOT_TOKEN, CHAT_ID, ip, new_otp)
            return "OTP expired. A new OTP has been sent."

        if otp_entered == correct_otp:
            os.system(f"sudo fail2ban-client set sshd unbanip {ip}")
            #del otp_store[ip]
            return "IP unblocked successfully!"

        return "Incorrect OTP."

    return render_template("otp.html", ip=ip)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
