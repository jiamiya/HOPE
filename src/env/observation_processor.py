import cv2
import matplotlib.pyplot as plt
import numpy as np

from configs import *

class Obs_Processor1():
    def __init__(self) -> None:
        self.downsample_rate = 4
        self.morph_kernel = np.array([
            [0,0,0,1,0,0,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [1,1,1,1,1,1,1],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,0,0,1,0,0,0]],dtype=np.uint8)
        self.n_channels = 2

    def process_img(self, img):
        obstacle_ege = self.get_obstacle_edge(img)
        img_traj = self.get_traj(img)

        return np.array([obstacle_ege, img_traj], dtype=np.float32).transpose(1,2,0)

    
    def rgb2binary(self, img, color):
        img_b = (img==color).astype(np.uint8)
        img_b = (np.sum(img_b,axis=-1) == 3).astype(np.uint8)
        # if self.n%50==0 :
        #     plt.imshow(img_b)
        #     plt.show()
        img_b = self.max_pooling2d(img_b)
        # img_b = cv2.resize(img_b, (img_b.shape[0]//self.downsample_rate,
        #     img_b.shape[1]//self.downsample_rate),interpolation= cv2.INTER_BITS) # INTERSECT_FULL
        # if self.n%50==0 :
        #     print(img_b.shape)
        #     plt.imshow(img_b)
        #     plt.show()
        return img_b
    
    def rgb2float(self, img, color_high, color_low):
        img_high = (img<=color_high).astype(np.uint8)
        img_high = (np.sum(img_high,axis=-1) == 3).astype(np.uint8)
        img_low = (img>=color_low).astype(np.uint8)
        img_low = (np.sum(img_low,axis=-1) == 3).astype(np.uint8)
        img_f = img_low*img_high
        img_f = self.max_pooling2d(img_f*cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        img_f = (img_f - img_f.min())/(img_f.max()+1e-8)
        return img_f

    
    def max_pooling2d(self, img:np.ndarray):
        w,h = img.shape
        k = self.downsample_rate
        if w%self.downsample_rate or h%self.downsample_rate:
            raise Warning('image shape can not be divided !')
        w_, h_ = w//k, h//k
        img = img.reshape(w_, k, h_, k).transpose(1,3,0,2).reshape(k**2, w_, h_)
        img = np.max(img, axis=0)
        return img


    def get_obstacle_edge(self, img):
        img_obstacle = self.rgb2binary(img, OBSTACLE_COLOR[:3])*255
        # plt.imshow(img_obstacle)
        # plt.show()
        img_obstacle = cv2.morphologyEx(img_obstacle, cv2.MORPH_CLOSE, self.morph_kernel)
        # plt.imshow(img_obstacle)
        # plt.show()
        # cv2.erode(img_obstacle)
        edge_obstacle = cv2.Canny(img_obstacle, 100, 200)
        # edge_obstacle = cv2.pyrDown(edge_obstacle,dstsize=(edge_obstacle.shape[0]//self.downsample_rate,
        #     edge_obstacle.shape[1]//self.downsample_rate))
        # edge_obstacle = cv2.resize(edge_obstacle, (edge_obstacle.shape[0]//self.downsample_rate,
        #     edge_obstacle.shape[1]//self.downsample_rate),interpolation= cv2.INTER_LINEAR)
        # print(edge_obstacle.shape)
        # print(edge_obstacle)
        # plt.imshow(edge_obstacle)
        # plt.show()
        # cv2.imshow('',img_obstacle)
        # cv2.waitKey(0)
        return edge_obstacle

    def get_traj(self, img):
        img_traj = self.rgb2float(img, TRAJ_COLOR_HIGH[:3], TRAJ_COLOR_LOW[:3])
        return img_traj
    
class Obs_Processor():
    def __init__(self) -> None:
        self.downsample_rate = 4
        self.n_channels = 3

    def process_img(self, img):
        # plt.imshow(img)
        # plt.show()
        processed_img = self.change_bg_color(img)
        # plt.imshow(processed_img)
        # plt.show()
        # p
        processed_img = cv2.resize(processed_img, (img.shape[0]//self.downsample_rate, img.shape[1]//self.downsample_rate))
        processed_img = processed_img/255.0

        return processed_img

    def change_bg_color(self, img):
        processed_img = img.copy()
        bg_pos = img==BG_COLOR[:3]
        bg_pos = (np.sum(bg_pos,axis=-1) == 3)
        processed_img[bg_pos] = (0,0,0)
        return processed_img
    
class Obs_Processor3():
    def __init__(self) -> None:
        self.downsample_rate = 4
        self.n_channels = 3

    def process_img(self, img):
        plt.imshow(img)
        plt.show()
        img_obstacle = self.rgb2binary(img, OBSTACLE_COLOR[:3])
        plt.imshow(img_obstacle)
        plt.show()
        img_dest = self.rgb2binary(img, DEST_COLOR[:3]) # TODO oclusion
        plt.imshow(img_dest)
        plt.show()
        img_traj = self.get_traj(img, TRAJ_COLOR_HIGH[:3], TRAJ_COLOR_LOW[:3])
        
        plt.imshow(img_traj)
        plt.show()

        return np.array([img_obstacle, img_dest, img_traj], dtype=np.float32).transpose(1,2,0)
    
    def get_traj(self, img:np.ndarray, color_high, color_low):
        # assume that the color high/low share the same color in the first 2 channels
        if color_high[:2] != color_low[:2]:
            raise NotImplementedError('color invalid')
        img_r, img_g, img_b = np.split(img, 3, axis=-1)
        img_r_1 = img_r==color_high[0]
        img_g_1 = img_g==color_high[1]
        img_b_1 = (img_b<=color_high[2]) * (img_b>=color_low[2])
        img_mask = img_r_1*img_g_1*img_b_1
        img_traj = (img_b-color_low[2]).astype(np.float32)/color_high[2] * img_mask
        return self.max_pooling2d(np.squeeze(img_traj))
        
    def rgb2binary(self, img, color):
        img_b = (img==color).astype(np.uint8)
        img_b = (np.sum(img_b,axis=-1) == 3).astype(np.uint8)
        img_b = self.max_pooling2d(img_b)
        return img_b
    
    def rgb2float(self, img, color_high, color_low):
        img_high = (img<=color_high).astype(np.uint8)
        img_high = (np.sum(img_high,axis=-1) == 3).astype(np.uint8)
        img_low = (img>=color_low).astype(np.uint8)
        img_low = (np.sum(img_low,axis=-1) == 3).astype(np.uint8)
        img_f = img_low*img_high
        img_f = self.max_pooling2d(img_f*cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        img_f = (img_f - img_f.min())/(img_f.max()+1e-8)
        return img_f

    
    def max_pooling2d(self, img:np.ndarray):
        w,h = img.shape
        k = self.downsample_rate
        if w%self.downsample_rate or h%self.downsample_rate:
            raise Warning('image shape can not be divided !')
        w_, h_ = w//k, h//k
        img = img.reshape(w_, k, h_, k).transpose(1,3,0,2).reshape(k**2, w_, h_)
        img = np.max(img, axis=0)
        return img