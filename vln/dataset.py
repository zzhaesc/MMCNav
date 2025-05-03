import os
import json
import numpy as np
import time
import re
import sys
import cv2
from PIL import Image
from vln.landmarks import filter_landmarks_5shot
from vln.mmc_env import get_closest_heading
from bokeh.plotting import figure
from bokeh.io.export import export_png
from bokeh.models import ColumnDataSource, LabelSet, Arrow, VeeHead

from tqdm import tqdm
import requests
import asyncio

def load_dataset(split, env, dataset_dir, dataset_name, caption_sets_dir, landmarks_file=None, size=-1):
    
    print('load ' + dataset_name + ' ' + split)
    with open(landmarks_file) as f:
        landmarks = json.load(f)['instances']

    instances = list()
    with open(os.path.join(caption_sets_dir, 'observation_llava-OV.json'), 'r', encoding='utf-8') as file:
            caption_sets = json.load(file)

    with open(os.path.join(dataset_dir, 'data',
                            f'{split}.json')) as f:
        for line in tqdm(list(f)):
            instance = dict(json.loads(line))  

            if dataset_name == 'touchdown':
                instance = preprocess_touchdown_instance(env, instance)

            idx = str(instance['id'])
            instance['idx'] = idx
            instance['dataset_name'] = dataset_name

            if idx not in landmarks:
                unfiltered = []
            else:
                unfiltered = landmarks[idx]['unfiltered']
            instance['landmarks'] = filter_landmarks_5shot(
                unfiltered)  # （intersection | side street | traffic lights | double lights | a building | a large building）

            instance['orig_route_panoids'] = instance['route_panoids']  # route_panoids can be overwritten by Dagger
            instance['is_novel'] = False
            instance['target_panoid'] = instance['route_panoids'][-1]
            instance['final_state'] = instance['target_panoid']
            instance['is_map2seq'] = dataset_name == 'map2seq'
            instances.append(instance)

            if size > 0 and len(instances) == size:
                break

    return instances, caption_sets

def preprocess_touchdown_instance(env, instance):
    instance['id'] = instance['route_id']
    start_pano = instance['route_panoids'][0] 
    start_node = env.graph.nodes[start_pano]
    start_heading = get_closest_heading(instance['start_heading'],
                                        start_node.neighbors.keys())
    instance['start_heading'] = start_heading

    return instance

def get_pano_slices(angles, fov, height, width, pano_image_path, pano_yaw, pano_heading):
    heading = pano_heading - pano_yaw
    heading = heading % 360

    slice_headings = [heading + angle for angle in angles]
    equ = Equirectangular(pano_image_path)
    images = list()

    if len(slice_headings) == 2:
        for i, slice_heading in enumerate(slice_headings):
            persp = equ.get_perspective(FOV=fov,
                                        THETA=slice_heading,
                                        PHI=3,
                                        height=height, 
                                        width=width)
            img = Image.fromarray(persp).convert('RGB')
            images.append(img)
        return images

    for i, slice_heading in enumerate(slice_headings):
        persp = equ.get_perspective(FOV=fov,
                                        THETA=slice_heading,
                                        PHI=3,
                                        height=height,  # 
                                        width=width)
        img = Image.fromarray(persp).convert('RGB')
        images.append(img)
    return images


class Equirectangular:
    # https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
    def __init__(self, img_name):
        im_cv = cv2.imread(img_name, cv2.IMREAD_ANYCOLOR)
        self._img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def get_perspective(self, FOV, THETA, PHI, height, width, RADIUS=128):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([height, width, 3], np.float32)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float32)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool_)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool_)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)
        return persp

            