import torch, torchvision
torchvision.disable_beta_transforms_warning()
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# INITIALIZE ------------------------------------------------------------------
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
img_path = "/home/chinchinati/Downloads/kitchen_PnP_Sink_Counter/processed_images/"
img_path = img_path + "frame_0.png"
image = Image.open(img_path)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.config.return_dict = False

# Visualize RAW image
# image.show()

# PREPROCESS IMAGE -------------------------------------------------------------
inputs = processor(images=image, return_tensors="pt")

# Visualize preprocessed image
# print(inputs["pixel_values"].shape) # ([1, 3, 224, 224])
# transform_to_pil = transforms.ToPILImage()
# pil_image = transform_to_pil(inputs["pixel_values"][0])
# pil_image.show()

# RUN MODEL -------------------------------------------------------------------
# outputs = model(**inputs)
# last_hidden_states = outputs[0]
# print(outputs[0].shape)

with torch.no_grad():
    traced_model = torch.jit.trace(model, [inputs.pixel_values])
    traced_outputs = traced_model(inputs.pixel_values)
print(traced_outputs[0].shape)
# print((last_hidden_states - traced_outputs[0]).abs().max())


# VISUALIZE EMBEDDINGS -------------------------------------------------------------
patch_embeddings = traced_outputs[0][0, 1:257-60, :]
patch_embeddings_norm = torch.norm(patch_embeddings, dim=-1)
print(patch_embeddings_norm.shape)
# Reshape to 14x14 (corresponding to the 16x16 patches in the image)
heatmap = patch_embeddings_norm.view(14, 14).detach().numpy()

# Plot the heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(image)
axes[0].axis('off')
axes[0].set_title('Original Image')

# Heatmap
im = axes[1].imshow(heatmap, cmap='viridis')
axes[1].axis('off')
axes[1].set_title('Patch Embedding Heatmap')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar to heatmap

# Show the plots
plt.tight_layout()
plt.show()