import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import random
import json
from PyWallpaper import change_wallpaper

def write_text(img, text, position, size):
    # Convert the image to RGB (OpenCV uses BGR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Pass the image to PIL
    pil_im = Image.fromarray(img)

    draw = ImageDraw.Draw(pil_im)
    # use a truetype font
    font1 = ImageFont.truetype("wallpaper/resources/Product Sans Regular.ttf", size)

    # Draw the text
    draw.text(position, text, font=font1)

    # Get back the image to OpenCV
    img = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)

    return img


def crop_circular_mask(image, background_color, diag_percent=1.56):
    roundmask = np.zeros(image.shape[:2], dtype='uint8')
    h, w = roundmask.shape[:2]
    diag = int(max(h, w)//diag_percent)
    roundmask = cv.circle(roundmask, (w//2, h//2), diag, (255, 255, 255), -1)

    image = cv.bitwise_and(image, image, mask=roundmask)
    black_pixels = np.where(
        (image[:, :, 0] == 0) &
        (image[:, :, 1] == 0) &
        (image[:, :, 2] == 0))

    # SET OFF BORDER COLORS TO BACKGROUND
    image[black_pixels] = background_color
    return image


config = None
with open('wallpaper/conf.json') as f:
    config = json.load(f)


template = cv.imread('wallpaper/template/CallTemplate.png')
unlock = None

call_name = config['custom_call_name']
if not call_name:
    call_name = os.getlogin()
# RELATIVE NAME POSITIONS
name_position = (60, 964)
miniature_pos = (1777, 42)  # X(1777:1801) Y(4s2:64)
unlock_pos = (349,1395)
guest_pos = (176,12)
unlock_writing_pos = (20,390)
miniature_resolution = (24, 24)

# BACKGROUND_COLORS
meet_background = [36, 33, 32]
chrome_backround = [59, 57, 51]

cap = cv.VideoCapture(0)
#cap = cv.VideoCapture(0, cv.CAP_DSHOW)

# cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

for i in range(3):
    ret, unlock = cap.read()

cap.release()

miniature = cv.imread('wallpaper/miniature/'+random.choice(os.listdir('wallpaper/miniature'))
                      ) if config['custom_miniature'] == True else unlock.copy()

random_guest = random.choice(os.listdir('wallpaper/guest'))
guest = cv.imread('wallpaper/guest/'+random_guest)



# POSITION TO REPLACE CALL IMAGE
desiredWidth = 510//2
desiredHeight = 424//2
height, width = unlock.shape[:2]
cx = width//2
cy = height//2

# RESIZE MINIATURE
miniature = cv.resize(miniature, miniature_resolution)
miniature = crop_circular_mask(miniature, chrome_backround, 2)

# RESIZE AND FIT guestIST IMAGE TO TEMPLATE
guest = cv.resize(guest, (1369, 780))
template[guest_pos[0]:(guest_pos[0]+guest.shape[0]), guest_pos[1]:(guest_pos[1]+guest.shape[1])] = guest

# CLEAR OLD NAMEPLATE FROM TEMPLATE IMAGE
nameplate = np.ones((20, 130, 3))
nameplate = nameplate*meet_background
template[969:(969+nameplate.shape[0]), 60:(60+nameplate.shape[1])] = nameplate

# crop call image to the right size
unlock = unlock[cy-desiredHeight:cy +
                desiredHeight, cx-desiredWidth:cx+desiredWidth]
unlock = crop_circular_mask(unlock, meet_background)

# WRITE NAMES INTO IMAGES
unlock = write_text(unlock, call_name, unlock_writing_pos, 16)
template = write_text(template, os.path.splitext(random_guest)[0], name_position, 22)

# FIT ACTUAL CALL INTO TEMPLATE
template[unlock_pos[0]:(unlock_pos[0]+unlock.shape[0]), unlock_pos[1]:(unlock_pos[1]+unlock.shape[1])] = unlock
template[miniature_pos[1]:(miniature_pos[1]+miniature.shape[0]),
         miniature_pos[0]:(miniature_pos[0]+miniature.shape[1])] = miniature

cv.imwrite('wallpaper/perdeu.png',template)
change_wallpaper(os.path.abspath('wallpaper/perdeu.png'))
#set_wallpaper(os.path.abspath('perdeu.png'))
#cv.imshow('Fonts', template)
#cv.waitKey(0)
