import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

HUMAN_DATASET_PATH = ["/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_navigate/NavigateKitchen/2024-05-09/demo_gentex_im128_randcams.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/multi_stage/chopping_food/ArrangeVegetables/2024-05-11/demo_im128.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/multi_stage/defrosting_food/MicrowaveThawing/2024-05-11/demo_im128.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/multi_stage/restocking_supplies/RestockPantry/2024-05-10/demo_im128.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/multi_stage/washing_dishes/PreSoakPan/2024-05-10/demo_im128.hdf5",
                        "/home/loki/robocasa/datasets/v0.1/multi_stage/brewing/PrepareCoffee/2024-05-07/demo_im128.hdf5"]
f = h5py.File(HUMAN_DATASET_PATH[1])
demo = f["data"]["demo_5"]                        # access demo 5
print(demo.keys())
obs = demo["obs"]                                 # obervations across all timesteps
left_img = obs['robot0_eye_in_hand_image']    # get left camera images in numpy format
# left_img = obs['robot0_eye_in_hand_image']    # get left camera images in numpy format
ep_meta = json.loads(demo.attrs["ep_meta"])       # get meta data for episode
lang = ep_meta["lang"]                            # get language instruction for episode
print(obs.keys())
# print(obs['object'])
print(obs['object'].fields)
print(obs['object'])
print(obs['object'].dtype)
object = obs['object']
print(object.shape)
breakpoint()

# log = pd.DataFrame(columns=["eef_x", "eef_y", "eef_z", "object_x", "object_y", "object_z", "diff_x", "diff_y", "diff_z", "diff_obj_x", "diff_obj_y", "diff_obj_z", "sum_x", "sum_y", "sum_z"])
log = pd.DataFrame(columns=["eef_x", "eef_y", "eef_z", "eff_qx", "eff_qy", "eff_qz", "eff_qw", "obj_x", "obj_y", "obj_z", "obj_qx", "obj_qy", "obj_qz", "obj_qw"])

# Initialize the plot
fig, ax = plt.subplots()
im = ax.imshow(left_img[0])
global itr, prev, sum, diff
prev = obs['object'][0]
sum = np.zeros_like(obs['object'][0])
diff = np.zeros_like(obs['object'][0])
itr = 0
# Update function for animation

def print_as_3f(obj_pose):
    ret = [f"{x:.3f}" for x in obj_pose]
    print(ret)

def update(frame):
    global itr, prev, sum, diff

    print(frame)

    x, y, z = obs['robot0_eef_pos'][frame]
    qx, qy, qz, qw = obs['robot0_eef_quat'][frame]
    obj_pose = obs['object'][frame]
    # diff_obj = obj_pose - prev
    # sum += abs(diff_obj)
    # prev = obj_pose
    diff = obj_pose[:3] - obs['robot0_eef_pos'][frame]

    # log.loc[frame] = [x, y, z, obj_pose[0], obj_pose[1], obj_pose[2], diff[0], diff[1], diff[2], diff_obj[0], diff_obj[1], diff_obj[2], sum[0], sum[1], sum[2]]
    log.loc[frame] = [x, y, z, qx, qy, qz, qw, obj_pose[0], obj_pose[1], obj_pose[2], obj_pose[3], obj_pose[4], obj_pose[5], obj_pose[6]]

    cost = np.linalg.norm(diff)
    print(f"Distance: {cost}")
    if cost < 0.05:
        plt.imsave("robocasa/temp/success.png", left_img[frame])
        exit()

    # print(x, y, z)
    # print_as_3f(obj_pose)
    # print_as_3f(diff)
    # print_as_3f(sum)

    # condition = frame < 50
    # # condition = frame > len(left_img) - 50
    # if condition:
    #     itr += 1
    #     im.set_array(left_img[frame-itr])
    #     # print(left_img[frame].shape)
    #     return [im]
    im.set_array(left_img[frame])
    # print(left_img[frame].shape)
    return [im]

ani = FuncAnimation(fig, update, frames=range(0, len(left_img)), interval=50, blit=False, repeat=False) 
plt.show()

# log.to_csv("robocasa/temp/log.csv")


f.close()