import processing
import os

base_dir = 'data_test'

for folder in os.listdir(base_dir):
    if folder.startswith('.'):
        continue
    for video in os.listdir(os.path.join(base_dir, folder)):
        if video.startswith('.'):
            continue
        print(f'Processing {video}')
        video_name = video.split('.')[0]
        video_path = os.path.join(base_dir, folder, video)
        if not os.path.exists(os.path.join(base_dir, folder, video_name)):
            os.mkdir(os.path.join(base_dir, folder, video_name))
        print(video_path)
        df = processing.video_to_df(video_path, sample_interval=7.5)
        processing.df_to_images(df, video_name, os.path.join(base_dir, folder, video_name))
        processing.df_to_csv(df, video_name, os.path.join(base_dir, folder, video_name))





