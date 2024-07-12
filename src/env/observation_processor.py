import cv2
import numpy as np

from configs import *
    
class Obs_Processor():
    def __init__(self) -> None:
        self.downsample_rate = 4
        self.n_channels = 3

    def process_img(self, img):
        processed_img = self.change_bg_color(img)
        processed_img = cv2.resize(processed_img, (img.shape[0]//self.downsample_rate, img.shape[1]//self.downsample_rate))
        processed_img = processed_img/255.0

        return processed_img

    def change_bg_color(self, img):
        processed_img = img.copy()
        bg_pos = img==BG_COLOR[:3]
        bg_pos = (np.sum(bg_pos,axis=-1) == 3)
        processed_img[bg_pos] = (0,0,0)
        return processed_img
    