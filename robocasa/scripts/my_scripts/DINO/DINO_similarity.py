import os
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Model and Processor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.config.return_dict = False  # Disable return_dict for tracing
model = model.half().to(device)  # Convert to FP16 and move to GPU

# Path to the folder containing images
# img_folder = "/home/chinchinati/Downloads/kitchen_PnP_counter_cabinet_full_traj/processed_images"
img_folder = "/home/chinchinati/Downloads/kitchen_PnP_cabinet_counter_full_traj/processed_images" # frame_85
img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')],
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

# Helper function to process image and extract embeddings
def extract_embeddings(img_path):
    # Load and preprocess the image
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs.pixel_values = inputs.pixel_values.half().to(device)  # Convert input to FP16 and move to GPU

    # Pass through the model
    with torch.no_grad():
        traced_outputs = model(inputs.pixel_values)

    # Extract patch embeddings
    patch_embeddings = traced_outputs[0][0, 1:257-60, :]  # Exclude [CLS] and extra tokens
    return patch_embeddings

# Define a target embedding from the first frame
target_image_path = img_files[85]
target_embeddings = extract_embeddings(target_image_path).mean(dim=0, keepdim=True)  # Average over patches

# Calculate cosine similarity for all frames
cosine_similarities = []
for img_path in img_files:
    frame_embeddings = extract_embeddings(img_path).mean(dim=0, keepdim=True)  # Average over patches
    similarity = torch.nn.functional.cosine_similarity(target_embeddings, frame_embeddings).item()
    cosine_similarities.append(similarity)

# Plot cosine similarity
plt.figure(figsize=(8, 4))
plt.plot(cosine_similarities, marker='o', linestyle='-', color='b')
plt.title('Cosine Similarity to Target Embedding Over Frames')
plt.xlabel('Frame Index')
plt.ylabel('Cosine Similarity')
plt.grid(True)
plt.tight_layout()
plt.show()
