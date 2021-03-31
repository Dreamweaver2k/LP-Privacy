import pytesseract
from PIL import Image
import cv2

def get_chars(image):
    text = ''
    text = pytesseract.image_to_string(image , lang='eng', config='--psm 7 --eom 3 -c tessedit_char_whitelist=ADF0123456789')
    text = text.strip()
    words = text.split(' ')
    plate = ''
    for w in words:
      if len(w) > len(plate): plate = w
    return plate

