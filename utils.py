import re
import os
import cv2
from sys import platform
if platform == "linux" or platform == "linux2":
    import tesseract as pytesseract
else:
    import pytesseract
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\tesseract\tesseract.exe'
from dateparser import parse

# Crop configs
LAT_CROP_CONFIG = {
    'left': 76,
    'right': 324,
    'top': 45,
    'bottom': 84
}

LON_CROP_CONFIG = {
    'left': 355,
    'right': 600,
    'top': 45,
    'bottom': 84
}

DATE_CROP_CONFIG = {
    'left': 1050,
    'right': 1225,
    'top': 45,
    'bottom': 84
}

# Preprocessing


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop(img, crop_config):
    return img[crop_config['top']:crop_config['bottom'], crop_config['left']:crop_config['right']]


def threshold(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def resize(img):
    return cv2.resize(img, (0, 0), fx=2, fy=2)


def preprocess(img, crop_config):
    gray = grayscale(img)
    cropped = crop(gray, crop_config)
    thresh = threshold(cropped)
    resized = resize(thresh)
    return resized

# Text Recognition


def recognize_text(img):
    #custom_config = r'-l eng --oem 3 --psm 7 -c tessedit_char_whitelist=0123456789\'\".°NESW/'
    custom_config = r'-l eng --oem 3 --psm 7 --dpi 70'
    # return pytesseract.image_to_string(img, config=custom_config).replace('\n', '').strip('\x0c')
    return pytesseract.image_to_string(img).replace('\n', '').strip('\x0c')


def dms_to_dd(dms):
    deg, minutes, seconds, direction = re.split('[°\'"]', dms)
    dd = (float(deg) + float(minutes) / 60 + float(seconds) /
          (60 * 60)) * (-1 if direction in ['W', 'S'] else 1)
    return round(dd, 6)


def parse_coord(coord_str):
    try:
        return dms_to_dd(coord_str.replace('\'\"', '\"'))
    except ValueError:
        print(f'Could not parse coord: {coord_str}', end="\r", flush=True)
        return None


def str_to_date(date_str):
    return parse(date_str, languages=['en'])


def parse_date(date_str):
    try:
        return str_to_date(date_str)
    except ValueError:
        print(f'Could not parse date: {date_str}', end="\r", flush=True)
        return None
