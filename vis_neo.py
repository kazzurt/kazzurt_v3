import board
import neopixel
pixels = neopixel.NeoPixel(board.D21,1300)
pixels.fill = ((0,0,0))
print("neopixel")
