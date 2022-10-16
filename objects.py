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
    return utils.preprocess(self.image, crop_config)

  def extract_lat(self):
    self.preprocessed_lat = self.preprocess(utils.LAT_CROP_CONFIG)
    return utils.parse_coord(utils.recognize_text(self.preprocessed_lat))

  def extract_lon(self):
    self.preprocessed_lon = self.preprocess(utils.LON_CROP_CONFIG)
    return utils.parse_coord(utils.recognize_text(self.preprocessed_lon))

  def extract_date(self):
    self.preprocessed_date = self.preprocess(utils.DATE_CROP_CONFIG)
    return utils.parse_coord(utils.recognize_text(self.preprocessed_date))

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
      i += 1
      date = Frame(self.get_frame(i)).extract_date()
    return date

  def frame_to_series(self, frame_idx):
    frame = Frame(self.get_frame(frame_idx))
    return pd.DataFrame({
        'Latitude': [frame.extract_lat()],
        'Longitude': [frame.extract_lon()],
        'Frame': [frame.image]
    })

  def extract_frames(self, sample_interval, retries=5):
    frame_interval = int(sample_interval * self.fps)
    n_frames = int(self.length / frame_interval)

    self.extracted_frames = []
    failed = 0
    duplicates = 0

    for i in range(0, self.length, frame_interval):
      print(
          f'Extracting frames: {int(i/frame_interval)+1}/{n_frames+1}', end="\r", flush=True)
      frame = Frame(self.get_frame(i)).extract()
      j = i
      while (frame.lat is None or frame.lon is None) and (j - i < retries) and (j < self.length):
        print(f'\n Text recognition failed, retrying {j-i+1}/{retries}')
        j += 1
        frame = Frame(self.get_frame(j)).extract()
        if frame.lat == None:
          cv2.imwrite(
              f'sample/frames/failed_lat{j}.png', frame.preprocessed_lat)
        if frame.lon == None:
          cv2.imwrite(
              f'sample/frames/failed_lon{j}.png', frame.preprocessed_lon)
      if frame.lat is None or frame.lon is None:
        failed += 1
        continue
      if len(self.extracted_frames) > 0:
        if (self.extracted_frames[-1].lat == frame.lat and
                self.extracted_frames[-1].lon == frame.lon):
          duplicates += 1
          continue
      self.extracted_frames.append(frame)

    print(
        f'\n Failed: {failed} ({int(failed/(n_frames+1)*100)}%) ; Duplicates: {duplicates} ({int(duplicates/(n_frames+1)*100)}%)')

  def to_dataframe(self, sample_interval):
    frame_interval = int(sample_interval * self.fps)
    n_frames = int(self.length / frame_interval)
    df = pd.DataFrame(columns=['Latitude', 'Longitude', 'Timestamp', 'Frame'])

    for i in range(0, self.length, frame_interval):
      print(
          f'Extracted frames: {int(i/frame_interval)+1}/{n_frames+1}', end="\r", flush=True)
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
      row['Frame'].generate_metadata(
          row['Latitude'], row['Longitude'], row['Timestamp'])
      row['Frame'].save_with_metadata(
          os.path.join(path, f'{self.video_name}_{ind}.jpg'))

  # Drop the frame column and save to disk
  def to_csv(self, path=os.getcwd()):
    self.df.drop(columns=['Frame'], inplace=True)
    self.df.to_csv(os.path.join(path, f'{self.video_name}.csv'), index=False)
