#!/usr/bin/env python3
print("hello there")

import RPi.GPIO as GPIO
boardRevision = GPIO.RPI_REVISION
GPIO.setmode(GPIO.BCM)

print(boardRevision)

