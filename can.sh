#!/bin/bash

sudo udevadm control --reload-rules && sudo udevadm trigger

sleep 1

sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up

echo "CAN setup complete"