import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from multiprocessing import Pool

HUMAN_DATASET_PATHS = ["/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5",
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

DURATION = 50

def plot_traj(path, display = False):
    dataset_dir = os.path.dirname(path)
    output_dir = os.path.join(dataset_dir, "processed_images")
    os.makedirs(output_dir, exist_ok=True)
    costs = []

    with h5py.File(path, "r") as f:
        demo = f["data"]["demo_5"]
        obs = demo["obs"]
        breakpoint()
        left_img = obs['robot0_eye_in_hand_image']

        for i in range(0, obs['object'].shape[1], 7):

            for frame in range(len(left_img)):  
                x, y, z = obs['robot0_eef_pos'][frame]
                obj_pose = obs['object'][frame]
                diff = obj_pose[i:i+3] - obs['robot0_eef_pos'][frame]
                cost = np.linalg.norm(diff)
                costs.append(cost)
                if display:
                    img = left_img[frame]
                    cv.putText(img, f"cost: {cost:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
                    cv.imshow("img", img)   
                    if cv.waitKey(DURATION) & 0xFF == ord('q'):
                        break 

            plt.plot(costs)
            plt.xlabel('frames')
            plt.ylabel('cost = norm(eef, obj)')
            plt.grid(True)
            plt.show()
    print(f"Processed dataset {path}")

def main():

    path = "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams.hdf5"

    plot_traj(path, display = True)
    
if __name__ == "__main__":
    main()
