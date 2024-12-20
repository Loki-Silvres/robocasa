import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
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

THRESHOLD = 0.00

def process_dataset(path):
    dataset_dir = os.path.dirname(path)
    output_dir = os.path.join(dataset_dir, "processed_images")
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(path, "r") as f:
        demo = f["data"]["demo_5"]
        obs = demo["obs"]
        left_img = obs['robot0_eye_in_hand_image']

        for frame in range(len(left_img)):
            x, y, z = obs['robot0_eef_pos'][frame]
            obj_pose = obs['object'][frame]
            diff = obj_pose[:3] - obs['robot0_eef_pos'][frame]

            cost = np.linalg.norm(diff)
            if cost > THRESHOLD:
                image_path = os.path.join(output_dir, f"frame_{frame}.png")
                plt.imsave(image_path, left_img[frame])

    print(f"Processed dataset {path}")

def main():
    # with Pool(processes=os.cpu_count()) as pool:
    #     pool.map(process_dataset, HUMAN_DATASET_PATHS)

    path = "/home/loki/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5"
    process_dataset(path)
    
if __name__ == "__main__":
    main()
