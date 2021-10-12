
import base64
import io
import os
import time

import cv2
from PIL import Image

from colorize import run

# Commonly used paths, let's define them here as constants
DATA_DIR_PATH = os.path.join(os.getcwd(), 'images')
INPUT_DATA_PATH = os.path.join(DATA_DIR_PATH, 'input')
OUT_IMAGES_PATH = os.path.join(DATA_DIR_PATH, 'output')

# Make sure these exist as the rest of the code relies on it
os.makedirs(OUT_IMAGES_PATH, exist_ok=True)
os.makedirs(INPUT_DATA_PATH, exist_ok=True)



# Callback function representing the main execution entry point
def execute(image):
    imageName = str(round(time.time() * 1000))
    base_dir = os.path.dirname(os.path.realpath(__file__))
    imagePath = os.path.join(base_dir + '/images/input')

    print("Start Coloring..........")
    print("imageName",imageName,type(imageName))
    image = image.replace('data:image/jpeg;base64,', '')
    image = bytes(image, 'utf-8')
    image = base64.b64decode(image)
    Image.open(io.BytesIO(image)).save(imagePath + f'/{imageName}.jpg')
    
    res = run(imageName)
    imageRGB = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(imageRGB)
    img.save(f"images/output/{imageName}.jpg")   

    imgList = []
    base_dir = os.path.dirname(os.path.realpath(__file__))
    outImagePath = os.path.join(base_dir + '/images/output/')
    with open(f"images/output/{imageName}.jpg", "rb") as image_file:
        data = base64.b64encode(image_file.read())
        imgList.append(b"data:image/jpeg;base64,"+data)
    print(imagePath)
    
    print(f"[Delete] {imagePath}/{imageName}.jpg")
    os.remove(f"{imagePath}/{imageName}.jpg")
    os.remove(f"{outImagePath}/{imageName}.jpg")
    print("[Done]")

    return imgList[0]

