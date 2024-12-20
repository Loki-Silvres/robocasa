import os
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Model and Processor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.config.return_dict = False  # Disable return_dict for tracing
model = model.half().to(device)  # Quantize to float16 and move to GPU

# Trace the model
dummy_inputs = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt").pixel_values.half().to(device)
with torch.no_grad():
    traced_model = torch.jit.trace(model, dummy_inputs)
print("Model tracing completed.")

# Path to the folder containing images
img_folder = "/home/chinchinati/Downloads/kitchen_PnP_cabinet_counter/processed_images/"
img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')],
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

# Helper function to process image and generate heatmap
def process_image(img_path):
    # Load and preprocess the image
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs.pixel_values = inputs.pixel_values.half().to(device)  # Convert inputs to FP16 and move to GPU

    # Pass through the traced model
    with torch.no_grad():
        traced_outputs = traced_model(inputs.pixel_values)

    # Extract patch embeddings
    patch_embeddings = traced_outputs[0][0, 1:257-60, :]  # Exclude [CLS] and extra tokens
    patch_embeddings_norm = torch.norm(patch_embeddings, dim=-1)

    # Reshape to 14x14
    heatmap = patch_embeddings_norm.view(14, 14).cpu().numpy()  # Move back to CPU for visualization
    return image, heatmap

# Animation setup
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image setup
image_ax = axes[0]
image_ax.axis('off')
image_ax.set_title('Original Image')

# Heatmap setup
heatmap_ax = axes[1]
heatmap_ax.axis('off')
heatmap_ax.set_title('Patch Embedding Heatmap')

# Initial processing for setup
initial_image, initial_heatmap = process_image(img_files[0])
image_plot = image_ax.imshow(initial_image)
heatmap_plot = heatmap_ax.imshow(initial_heatmap, cmap='viridis')
colorbar = fig.colorbar(heatmap_plot, ax=heatmap_ax, fraction=0.046, pad=0.04)  # Add static colorbar

# Function to update frames
def update(frame_idx):
    img_path = img_files[frame_idx]
    image, heatmap = process_image(img_path)

    # Update original image
    image_plot.set_data(image)

    # Update heatmap data
    heatmap_plot.set_data(heatmap)

    # Update titles
    image_ax.set_title(f'Original Image: {os.path.basename(img_path)}')

# Create animation
anim = FuncAnimation(fig, update, frames=len(img_files), interval=1, repeat=True)  # ~30 FPS (1000 ms / 30 ≈ 33 ms)
# anim.save('Embedding_Visualization.mp4', writer='ffmpeg', fps=30)  # Save as MP4 with 30 FPS

plt.tight_layout()
plt.show()
