from PIL import Image
import os

# Define path within your MapGPT structure
assets_dir = "figs" # Assuming MapGPT is the current working directory
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)
placeholder_path = os.path.join(assets_dir, "placeholder_pruned.png")

img = Image.new('RGB', (1, 1), color = 'grey')
img.save(placeholder_path)
print(f"Placeholder image saved to {os.path.abspath(placeholder_path)}")