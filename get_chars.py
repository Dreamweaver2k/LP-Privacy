import pytesseract
from PIL import Image

def get_chars(images):
    text = ''
    for im in images:
        letter = pytesseract.image_to_string(im, config='--psm 10 --oem 6 -c tessedit_char_whitelist=ABCDEFG123456789')
        if letter == 'l': letter = '1'
        text += letter.strip()
    return text
