import os
import cv2
import utils
import pandas as pd
from GPSPhoto import gpsphoto

class Frame:
  def __init__(self, image):
    self.image = image

  def extract(self):
    self.lat = self.extract_lat()
    self.lon = self.extract_lon()
    return self
  
  def preprocess(self, crop_config):
    self.preprocessed = utils.preprocess(self.image, crop_config)
    return self
  
  def recognize_text(self):
    return utils.recognize_text(self.preprocessed)
  
  def extract_lat(self):
    return utils.parse_coord(self.preprocess(utils.LAT_CROP_CONFIG).recognize_text())
  
  def extract_lon(self):
    return utils.parse_coord(self.preprocess(utils.LON_CROP_CONFIG).recognize_text())

  def extract_date(self):
    return utils.parse_date(self.preprocess(utils.DATE_CROP_CONFIG).recognize_text())
  
  def generate_metadata(self, lat, lon, date):
    self.metadata = gpsphoto.GPSInfo((lat, lon), timeStamp=date)

  def save_with_metadata(self, path):
    cv2.imwrite(path, self.image)
    photo = gpsphoto.GPSPhoto(path)
    photo.modGPSData(self.metadata, path)

class Video:
  def __init__(self, path):
    self.path = path
    self.video_name = os.path.basename(path)
    self.cap = cv2.VideoCapture(path)
    self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

  def get_frame(self, frame_idx):
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = self.cap.read()
    return frame
  
  def extract_date(self):
    i = 0
    date = Frame(self.get_frame(i)).extract_date()
    while date is None:
      i+=1
      date = Frame(self.get_frame(i)).extract_date()
    return date
  
  def frame_to_series(self, frame_idx):
    frame = Frame(self.get_frame(frame_idx))
    return pd.DataFrame({
      'Latitude': [frame.extract_lat()],
      'Longitude': [frame.extract_lon()],
      'Frame': [frame.image]
    })
  
  def extract_frames(self, sample_interval):
    frame_interval = int(sample_interval * self.fps)
    n_frames = int(self.length/frame_interval)

    self.extracted_frames = []
    failed = 0
    duplicates = 0

    for i in range(0, self.length, frame_interval):
      print(f'Extracting frames: {int(i/frame_interval)+1}/{n_frames+1}', end="\r", flush=True)
      frame = Frame(self.get_frame(i)).extract()
      cv2.imwrite(f'sample/frames/frame{i}.jpg',frame.preprocessed)
      if frame.lat is not None and frame.lon is not None:
        for extracted_frame in self.extracted_frames:
          if extracted_frame.lat == frame.lat and extracted_frame.lon == frame.lon:
            duplicates+=1
            break
      else:
        failed+=1
        continue
      self.extracted_frames.append(frame)

    #print(f'\n Successfully extracted {len(self.extracted_frames)}/{n_frames+1} frames')
    print(f'\n Failed: {failed} ({int(failed/(n_frames+1)*100)}%) ; Duplicates: {duplicates} ({int(duplicates/(n_frames+1)*100)}%)')


  def to_dataframe(self, sample_interval):
    frame_interval = int(sample_interval * self.fps)
    n_frames = int(self.length/frame_interval)
    df = pd.DataFrame(columns=['Latitude', 'Longitude', 'Timestamp','Frame'])

    for i in range(0, self.length, frame_interval):
      print(f'Extracted frames: {int(i/frame_interval)+1}/{n_frames+1}', end="\r", flush=True)
      df = pd.concat([df, self.frame_to_series(i)], ignore_index=True)
    
    length = len(df)
    print(length)

    df.dropna(inplace=True)
    print(df)
    # df = df.drop_duplicates(subset=['Latitude', 'Longitude'])
    # print(f'\n Removed {length - len(df)} duplicate frames')

    df['Timestamp'] = self.extract_date()

    # Drop frame column
    print(df.drop(columns=['Frame'], axis=1))

    self.df = df
    return self

  # Save extracted frames to disk
  def to_images(self, path=os.getcwd()):
    for ind, row in self.df.iterrows():
      row['Frame'].generate_metadata(row['Latitude'], row['Longitude'], row['Timestamp'])
      row['Frame'].save_with_metadata(os.path.join(path, f'{self.video_name}_{ind}.jpg'))
  
  #Drop the frame column and save to disk
  def to_csv(self, path=os.getcwd()):
    self.df.drop(columns=['Frame'], inplace=True)
    self.df.to_csv(os.path.join(path, f'{self.video_name}.csv'), index=False)