import cv2

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




