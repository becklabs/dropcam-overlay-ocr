import os
import re
import cv2
import preprocessing
import crop_configs
import pytesseract
if os.name == 'nt':
  pytesseract.pytesseract.tesseract_cmd = r'C:\tesseract\tesseract.exe'
import pandas as pd
from dateparser import parse
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
    n_frames = int(length/frame_interval)

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
                df = pd.concat([df, pd.DataFrame(
                  {'Latitude': [lat],
                 'Longitude': [lon],
                  'Timestamp': [date],
                  'Frame': [frame]})],
                   ignore_index=True)
                last_lat = lat
                last_long = lon
            print(f'{int(i/frame_interval)+1}/{n_frames+1}')
    cap.release()
    return df

def generate_metadata(lat, lon, date):
    return gpsphoto.GPSInfo((lat, lon), timeStamp=date)

def add_metadata(frame_path, metadata):
    photo = gpsphoto.GPSPhoto(frame_path)
    photo.modGPSData(metadata, frame_path)

def df_to_images(df, photo_prefix='', output_folder=''):
    for ind, row in df.iterrows():
        frame_path = f'{output_folder}/{photo_prefix}_{ind}.jpg'
        cv2.imwrite(frame_path, row['Frame'])
        metadata = generate_metadata(row['Latitude'], row['Longitude'], row['Timestamp'])
        add_metadata(frame_path, metadata)

def video_to_geotagged_images(video_path, output_path=False, sample_interval=7.5, export_csv=False):
  """
  Extracts the coords from the video and saves the images with the GPS coordinates as metadata

  Arguments:
  video_path: Path to video file
  output_path: where the images and csv will be exported
  sample_interval: interval in seconds between each image

  """
  # (Ex. if prefix = "sample": export is sample_0.jpg, sample_1.jpg ... )
  photo_prefix = os.path.basename(video_path).split('.')[0]

  # Create output folder if it doesn't exist
  if not output_path:
    output_path = os.path.join(os.path.dirname(video_path), photo_prefix)
    
  if not os.path.exists(output_path):
      os.makedirs(output_path)
      
  # Extract frames from video to a table with Lat, Lon, Timestamp, Image (dataframe)
  df = video_to_df(video_path=video_path, sample_interval=7.5)

  # Save each frame in a table along with its respective EXIF data
  df_to_images(df=df, photo_prefix=photo_prefix,
             output_folder=output_path)

  # Drop the 'Frame' column and save table as csv
  if export_csv:
    print('Saving datatable...')
    output_df = df.drop(columns=['Frame'])
    output_df.to_csv(os.path.join(output_path, photo_prefix) + '.csv', index=False)

  print('Done')
