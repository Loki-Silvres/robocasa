import h5py
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
f = h5py.File(HUMAN_DATASET_PATH[28])
demo = f["data"]["demo_5"]                        # access demo 5
# print(demo.keys())
obs = demo["obs"]                                 # obervations across all timesteps
left_img = obs['robot0_agentview_left_image']    # get left camera images in numpy format
in_hand_img = obs['robot0_eye_in_hand_image']    # get left camera images in numpy format
right_img = obs['robot0_agentview_right_image']    # get left camera images in numpy format
ep_meta = json.loads(demo.attrs["ep_meta"])       # get meta data for episode
lang = ep_meta["lang"]                            # get language instruction for episode
# print(obs.keys())
# print(obs['object'])

print('f.keys()', f.keys())
print()
print('f["data"].keys()', f["data"].keys())
print()
print('f["mask"].keys()', f["data"].keys())
print()
print('demo.keys()', demo.keys())
print()
print('obs.keys()', obs.keys())
print()
print(obs['object'].shape)

# for ep in f["data"].keys():
#     ep_meta = json.loads(f["data/{}".format(ep)].attrs["ep_meta"])
#     print(ep_meta["lang"])

# Initialize the plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
im_left = ax[0].imshow(left_img[0])
ax[0].set_title("Left Camera View")
im_in_hand = ax[1].imshow(in_hand_img[0])
ax[1].set_title("In-Hand Camera View")
im_right = ax[2].imshow(right_img[0])
ax[2].set_title("Right Camera View")

for axis in ax:
    axis.axis('off')

# Update function for animationz
def update(frame):
    im_left.set_array(left_img[frame])
    im_in_hand.set_array(in_hand_img[frame])
    im_right.set_array(right_img[frame])
    return [im_left, im_in_hand, im_right]

# Create the animation
ani = FuncAnimation(fig, update, frames=len(left_img), interval=10, blit=False)  # interval in ms
ani.save("robocasa/temp/animation.gif", writer="pillow") # to save as gif
plt.tight_layout()
# Display the animation
plt.show()


f.close()