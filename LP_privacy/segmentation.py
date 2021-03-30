import cv2


def segmentation(img):
    img = cv2.equalizeHist(img)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 65, 15)
    img_edges = cv2.Canny(img, 0,250)
    contours, h = cv2.findContours(img_edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #change to contours, h
    characters = []
    for segment in contours:
        xmin = float('inf')
        xmax = 0
        ymin = float('inf')
        ymax = 0
        for val in segment:
            if xmin > val[0][0]:
                xmin = val[0][0]
            elif xmax < val[0][0]:
                xmax = val[0][0]
            if ymin > val[0][1]:
                ymin = val[0][1]
            elif ymax < val[0][1]:
                ymax = val[0][1]
        xdiff = xmax - xmin
        ydiff = ymax - ymin
        height, width = img.shape
        buffer = 5


        if xdiff < .2 * width and ydiff > .5 * height:
            characters.append([xmin, img[max(ymin-buffer,0):min(ymax+buffer, height), max(xmin-buffer,0):min(xmax+buffer,width)]])
    characters = sorted(characters)
    keep = []
    for i in range(1,len(characters)):
        if i < len(characters) - 1:
            if abs(characters[i][0] - characters[i-1][0]) > .04 * width:
                keep.append(i)
        else:
            keep.append(i)
    character_im = []
    for i in range(len(characters)):
        if i in keep:
            new_im = cv2.resize(characters[i][1], (100, 150))
            #ret, thresh_im = cv2.threshold(new_im, 90, 255, cv2.THRESH_BINARY)
            character_im.append(new_im)
    return character_im

    cv2.waitKey()