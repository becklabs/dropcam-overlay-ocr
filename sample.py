import pandas as pd
from processing import video_to_df, df_to_images

####################################
# Path to video file
VIDEO_PATH = 'sample_video.mp4'

# Photo prefix (Ex. if prefix = "sample": export is sample_0.jpg, sample_1.jpg ... )
# csv also named using prefix
PHOTO_PREFIX = "sample"

# Output folder: where the images and csv will be exported
OUTPUT_FOLDER = 'sample_video_output/'

####################################

# Extract frames from video to a table with Lat, Lon, Timestamp, Image (dataframe)
# Samples video at a given interval (seconds)
print('Extracting frames from video...')
df = video_to_df(video_path=VIDEO_PATH, sample_interval=7.5)

# Save each frame in a table along with its respective EXIF data
print('Saving Frames...')
df_to_images(df=df, photo_prefix=PHOTO_PREFIX,
             output_folder=OUTPUT_FOLDER)

# Drop the 'Frame' column and save table as csv
print('Saving datatable...')
output_df = df.drop(columns=['Frame'])
output_df.to_csv(OUTPUT_FOLDER + PHOTO_PREFIX + '.csv', index=False)

print('Done')
