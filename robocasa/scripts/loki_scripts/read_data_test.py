import h5py
import json
import numpy as np
import cv2 as cv

DATASET_PATH = '/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5'

f = h5py.File(DATASET_PATH)
demo = f["data"]["demo_5"]                        # access demo 5
obs = demo["obs"]                                 # obervations across all timesteps
left_img = obs["robot0_agentview_left_image"][:]  # get left camera images in numpy format
ep_meta = json.loads(demo.attrs["ep_meta"])       # get meta data for episode
lang = ep_meta["lang"]                            # get language instruction for episode

print(*[type(i) for i in [demo, obs, left_img, ep_meta, lang]])
while cv.waitKey(1) != ord("q"):
    h,w,_ = left_img[0].shape
    img = cv.resize(left_img[0], (w*4, h*4))
    cv.imshow("img", img)
f.close()