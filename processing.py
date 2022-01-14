import re
import cv2
import preprocessing
import crop_configs
import pytesseract
import pandas as pd
from dateparser import parse
import pytessy
from GPSPhoto import gpsphoto
import time

def first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame

def dms_to_dd(dms):
    deg, minutes, seconds, direction = re.split('[°\'"]', dms)
    dd = (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)
    return round(dd, 6)

def str_to_date(date_str):
    return parse(date_str, languages=['en'])

def recognize_text(img):
    #custom_config = r'-l eng --oem 3 --psm 7 -c tessedit_char_whitelist=0123456789\'\".°NESW/'
    custom_config = r'-l eng --oem 3 --psm 7'
    #return pytesseract.image_to_string(img, config=custom_config).replace('\n', '')
    return pytesseract.image_to_string(img, config=custom_config).replace('\n', '').strip('\x0c')

def process_coord(img, crop_config):
    coord = preprocessing.preprocess(img, crop_config)
    coord_str = recognize_text(coord).replace('\'\"', '\"')
    try:
        return dms_to_dd(coord_str)
    except ValueError:
        print(coord_str)
        return None

def process_lat(img):
    return process_coord(img, crop_configs.LAT_CROP_CONFIG)

def process_lon(img):
    return process_coord(img, crop_configs.LON_CROP_CONFIG)

def process_date(img):
    date = preprocessing.preprocess(img, crop_configs.DATE_CROP_CONFIG)
    date_str = recognize_text(date)
    try: 
        return str_to_date(date_str)
    except ValueError:
        print(date_str)
        return None

def extract_date(video_path):
    frame = first_frame(video_path)
    date = process_date(frame)
    return date

def video_to_df(video_path, sample_interval=5):

    df = pd.DataFrame(columns=['Latitude', 'Longitude', 'Timestamp','Frame'])

    date = extract_date(video_path)
    
    last_lat = None
    last_long = None

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(sample_interval * fps)

    for i in range(0, length, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            lat = process_lat(frame)
            lon = process_lon(frame)
            if lat == None or lon == None:
                print(f'Error at frame {i}')
                continue
            if lat != last_lat or lon != last_long:
                df = df.append({'Latitude': lat,
                 'Longitude': lon,
                  'Timestamp': date,
                  'Frame': frame}, ignore_index=True)
                last_lat = lat
                last_long = lon
            print(f'{i}/{length}')
    cap.release()
    return df

def generate_metadata(lat, lon, date):
    return gpsphoto.GPSInfo((lat, lon), timeStamp=date)

def add_metadata(frame_path, metadata):
    photo = gpsphoto.GPSPhoto(frame_path)
    photo.modGPSData(metadata, frame_path)

def df_to_images(df, video_name='', output_dir=''):
    for ind, row in df.iterrows():
        frame_path = f'{output_dir}/{video_name}_{ind}.jpg'
        cv2.imwrite(frame_path, row['Frame'])
        metadata = generate_metadata(row['Latitude'], row['Longitude'], row['Timestamp'])
        add_metadata(frame_path, metadata)

def df_to_csv(df, video_name='', output_dir=''):
    df = df.drop(columns=['Frame'])
    df.to_csv(f'{output_dir}/{video_name}.csv')


df = video_to_df('data/test2.mp4',sample_interval=7.5)
df_to_images(df, video_name='test2', output_dir='data/test2')

