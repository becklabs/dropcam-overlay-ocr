import cv2
import processing
import pandas as pd
from datetime import datetime

def extract_date(video_path):
    frame = processing.first_frame(video_path)
    date = processing.process_date(frame)
    return date

def video_to_df(video_path):
    now = datetime.now()
    df = pd.DataFrame(columns=['lat', 'lon', 'date'])

    date = extract_date(video_path)
    
    last_lat = None
    last_long = None

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    for i in range(0, length, fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            try:
                lat = processing.process_lat(frame)
                lon = processing.process_lon(frame)
            except ValueError:
                continue
            if lat != last_lat or lon != last_long:
                df = df.append({'lat': lat, 'lon': lon, 'date': date}, ignore_index=True)
                last_lat = lat
                last_long = lon
            print(f'{i}/{length}')
    cap.release()
    print(datetime.now() - now)
    return df

data = video_to_df('data/test.mp4')
#data.to_csv('data/test.csv')






