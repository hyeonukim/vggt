import torch
import numpy as np
import matplotlib.pyplot as plt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os

def main():
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    print(f">>> Running on: {device}")

    # 2. Load Model (Using 'S' for speed/memory, change to '1B' for max quality)
    print(">>> Loading Model (this may take a minute)...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # 3. Load Your Images
    # Replace these names with your actual filenames
    image_names = ["buttjoint_000115.png", "buttjoint_000215.png", "buttjoint_000255.png"]
    
    # Verify files exist before starting
    for img in image_names:
        if not os.path.exists(img):
            print(f"ERROR: Could not find {img} in the current folder!")
            return

    images = load_and_preprocess_images(image_names).to(device)

    # 4. Run Inference
    print(">>> Processing images...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
            predictions = model(images)

# 5. Extract Results with correct dimension handling
    # VGGT-1B outputs are typically [Batch, Channel, Height, Width]
    # For depth, it might be [Batch, 1, H, W] or [Batch, H, W]
    depths = predictions['depth'].cpu().numpy()              
    pts3d = predictions['world_points'].cpu().numpy()        
    
    # Process colors (convert from Torch [B, 3, H, W] to Numpy [B, H, W, 3])
    # permute(0, 2, 3, 1) means: Batch stays 0, H moves to 1, W moves to 2, C moves to 3
    colors = images.permute(0, 2, 3, 1).cpu().numpy()        

    # 6. Show and Save Depth Maps
# 6. Show and Save Depth Maps
    print(">>> Generating Depth Maps...")
    for i in range(len(depths)):
        plt.figure(figsize=(10, 5))
        
        # d_map shape starts as (3, 392, 518, 1)
        d_map = depths[i]
        
        # 1. Remove that trailing '1' -> result: (3, 392, 518)
        d_map = np.squeeze(d_map)
        
        # 2. If it's 3-channel (C, H, W), move the channels to the end -> (392, 518, 3)
        if d_map.ndim == 3 and d_map.shape[0] == 3:
            d_map = np.transpose(d_map, (1, 2, 0))
            # 3. Take the first channel for a 2D depth visualization
            d_map = d_map[:, :, 0]

        plt.imshow(d_map, cmap='magma')
        plt.title(f"Depth Map: {image_names[i]}")
        plt.axis('off')
        plt.savefig(f"depth_result_{i}.png")
        plt.show()
    # 7. Export Colored 3D Point Cloud (.ply)
    print(">>> Exporting 3D Point Cloud...")
    # We flatten the arrays to a list of points
    flat_pts = pts3d.reshape(-1, 3)
    flat_colors = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)

    # Optional: If the file is too big, only save every 4th point
    # flat_pts = flat_pts[::4]
    # flat_colors = flat_colors[::4]

    with open("buttjoint_reconstruction.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(flat_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(flat_pts, flat_colors):
            # Filtering out points at 0,0,0 which are usually empty space
            if np.all(p == 0): continue 
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    print("\nSUCCESS! Open 'buttjoint_reconstruction.ply' in an online 3D viewer.")
if __name__ == "__main__":
    main()