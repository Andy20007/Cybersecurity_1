#!/bin/bash

# Run forever
while true; do
    # Clear the log file
    sudo truncate -s 0 /var/log/auth.log

    # Wait for 30 seconds before repeating
    sleep  20
done
