import re
import cv2
import preprocessing
import crop_configs
import pytesseract
from dateparser import parse

def recognize_text(img):
    custom_config = r'-l eng --oem 3 --psm 7 -c tessedit_char_whitelist=0123456789\'\".°NESW/'
    return pytesseract.image_to_string(img, config=custom_config)

def dms_to_dd(dms):
    deg, minutes, seconds, direction =  re.split('[°\'"]', dms)
    return (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)

def str_to_date(date_str):
    return parse(date_str, languages=['en'])

def process_lat(img):
    lat = preprocessing.preprocess(img, crop_configs.LAT_CROP_CONFIG)
    lat_str = recognize_text(lat)
    #print(lat_str)
    return dms_to_dd(lat_str)

def process_lon(img):
    lon = preprocessing.preprocess(img, crop_configs.LON_CROP_CONFIG)
    lon_str = recognize_text(lon)
    return dms_to_dd(lon_str)

def process_date(img):
    date = preprocessing.preprocess(img, crop_configs.DATE_CROP_CONFIG)
    date_str = recognize_text(date)
    return str_to_date(date_str)

def first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame
