
import cv2
import numpy as np

#https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format
def transform(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = cv2.cornerHarris(img, 4, 7, 0.08)
    im = np.where(im > .07, 1.0, 0)
    tl_best = float('inf')
    tl_point = [0,0]
    tr_best = float('inf')
    tr_point = [im.shape[1],0]
    bl_best = float('inf')
    bl_point = [0, im.shape[0]]
    br_best = float('inf')
    br_point = [im.shape[1], im.shape[0]]

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] == 1:
                tl = i**2 + j**2
                tr = (im.shape[1] - j)**2 + i**2
                bl = j**2 + (im.shape[0]-i)**2
                br = (im.shape[0]-i)**2 + (im.shape[1] - j)**2
                best = min(tl,tr,bl,br)
                if tl == best and tl_best > best:
                    tl_best = best
                    tl_point = [j,i]
                elif tr == best and tr_best > best:
                    tr_best = best
                    tr_point = [j,i]
                elif bl == best and bl_best > best:
                    bl_best = best
                    bl_point = [j,i]
                elif br == best and br_best > best:
                    br_best = best
                    br_point = [j,i]

    points = np.float32([tl_point, tr_point, bl_point,  br_point])
    dest_points = np.float32([[0,0], [im.shape[1],0], [0, im.shape[0]], [im.shape[1], im.shape[0]]])
    if tr_best+tl_best+br_best+bl_best != float('inf'):
        A = cv2.getPerspectiveTransform(points, dest_points)
        img = cv2.warpPerspective(img, A, (im.shape[1],im.shape[0]))
    return img

