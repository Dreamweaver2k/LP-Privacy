import rsa
import time
import cv2
import numpy as np
import random


def encrypt(im, shuffle_num):
    (pubkey, privatekey) = rsa.newkeys(124)
    # im = cv2.imread('lp_buffer4.jpg')
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2 = im.copy()

    im2 = cv2.resize(im2, (160, 48))
    im3 = im2.copy()
    height, width, _ = im2.shape
    shufflekey = "{0:b}".format(shuffle_num)
    message = im[0][0].tobytes()

    enc = rsa.encrypt(message, pubkey)
    t = time.time()
    # for j in range(1000):
    zero = False
    one = False
    swap_x1 = 0
    swap_y1 = 0
    swap_x0 = 0
    swap_y0 = 0

    for x in range(0, height, 8):
        for y in range(0, width, 8):
            bit = int(shufflekey[int((x / 8 + y / 8))])
            if bit == 0 and not zero:
                zero = True
                swap_x0 = x
                swap_y0 = y

            elif bit == 1 and not one:
                one = True
                swap_x1 = x
                swap_y1 = y

            elif bit == 0 and zero:
                im2[x: x + 8, y: y + 8], im2[swap_x0: swap_x0 + 8, swap_y0: swap_y0 + 8] = im3[
                                                                                           swap_x0: swap_x0 + 8,
                                                                                           swap_y0: swap_y0 + 8], im3[
                                                                                                                  x: x + 8,
                                                                                                                  y: y + 8]
                zero = False
            elif bit == 1 and one:
                im2[x: x + 8, y: y + 8], im2[swap_x1: swap_x1 + 8, swap_y1: swap_y1 + 8] = im3[
                                                                                           swap_x1: swap_x1 + 8,
                                                                                           swap_y1: swap_y1 + 8], im3[
                                                                                                                  x: x + 8,
                                                                                                                  y: y + 8]
                one = False

    return im2

