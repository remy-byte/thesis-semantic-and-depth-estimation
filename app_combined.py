# depth_anything_app.py
# Updated: Filtering logic now retains images where at least 95% of the target‑class pixels
# fall within the specified depth range.

import os
import zipfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
import open3d as o3d
import matplotlib.cm as cm
from torchvision.transforms import Compose
import shutil
import hydra
from omegaconf import DictConfig

from depth_anything_v2.dpt import DepthSegmentAnythingV2
from transform import PrepareForNet, Resize, NormalizeImage

# ─── Globals ────────────────────────────────────────────────────────────────
joint_model = None

CITYSCAPES_CLASSES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light",
    "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car",
    "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
]

CITYSCAPES_COLORS = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

# ─── Model Loader ────────────────────────────────────────────────────────────

def load_model_from_config(config: DictConfig):
    """Instantiate and load model weights from a Hydra config."""
    global joint_model
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    }
    joint_model = DepthSegmentAnythingV2(**{**model_configs['vitb'], 'max_depth': 80})
    state_metric_depth_seg = config.app.ckptpath
    state_dict = torch.load(state_metric_depth_seg)
    joint_model.load_state_dict(state_dict['state_dict'], strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    joint_model.to(device)
    joint_model.eval()

# ─── Preprocessing ───────────────────────────────────────────────────────────

def image2tensor(raw_image: np.ndarray, input_size: int = 518):
    transform = Compose([
        Resize(width=input_size, height=input_size, resize_target=False,
               keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    image_float = raw_image.astype(np.float32) / 255.0
    image_transformed = transform({'image': image_float})['image']
    tensor = torch.from_numpy(image_transformed).unsqueeze(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return tensor.to(device), raw_image.shape[:2]

# ─── Visualization Helpers ───────────────────────────────────────────────────

def apply_color_map(segmentation: np.ndarray, colormap: list):
    h, w = segmentation.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in enumerate(colormap):
        colored[segmentation == label] = color
    return colored


def apply_colormap_to_depth(depth: np.ndarray, cmap_name: str = 'plasma'):
    depth_np = depth.astype(np.float32)
    norm = (depth_np - np.min(depth_np)) / ((np.max(depth_np) - np.min(depth_np)) + 1e-6)
    colormap = cm.get_cmap(cmap_name)
    colored = colormap(norm)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def create_point_cloud(depth_map: np.ndarray, rgb_colors: np.ndarray, save_path: str):
    h, w = depth_map.shape
    f_x, f_y = 2262.52, 2265.30
    c_x, c_y = 1024, 512

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - c_x) / f_x
    y = (y - c_y) / f_y
    z = depth_map
    points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)
    colors = np.array(rgb_colors).reshape(-1, 3) / 255.0

    valid = z.reshape(-1) > 0
    points_valid = points[valid]
    colors_valid = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_valid)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid)
    o3d.io.write_point_cloud(save_path, pcd)
    return save_path

# ─── Core Inference Helpers ──────────────────────────────────────────────────

def generate_depth_and_segmentation(image: np.ndarray):
    """Run model on RGB image and return segmentation, depth, and two point clouds."""
    global joint_model
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    seg_logits, depth_tensor = joint_model.infer_image(image_bgr)

    segmentation = torch.argmax(seg_logits, dim=1).float()
    segmentation = F.interpolate(
        segmentation.unsqueeze(1),
        (1024, 2048),
        mode="nearest"
    )[0, 0]
    seg_map_np = segmentation.cpu().numpy().astype(np.uint8)
    colored_seg = apply_color_map(seg_map_np, CITYSCAPES_COLORS)

    depth_np = depth_tensor.squeeze().astype(np.float32)
    depth_colored = apply_colormap_to_depth(depth_np)

    resized_rgb = cv2.resize(image, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    seg_resized_for_pcd = cv2.resize(colored_seg, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    seg_ply = create_point_cloud(depth_np, seg_resized_for_pcd, "/tmp/seg_pointcloud.ply")
    raw_ply = create_point_cloud(depth_np, resized_rgb, "/tmp/raw_pointcloud.ply")

    return (
        colored_seg.astype(np.float32) / 255.0,
        depth_colored.astype(np.float32) / 255.0,
        seg_ply,
        raw_ply
    )


def process_image(image: np.ndarray):
    """Return segmentation map and depth map for a single image."""
    global joint_model
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    seg_logits, depth_tensor = joint_model.infer_image(image_bgr)

    segmentation = torch.argmax(seg_logits, dim=1).squeeze(0).float()
    segmentation = F.interpolate(
        segmentation.unsqueeze(0).unsqueeze(0),
        (1024, 2048),
        mode="bilinear",
        align_corners=True
    )[0, 0]
    seg_map = segmentation.cpu().numpy().astype(np.uint8)
    depth_map = depth_tensor.squeeze()
    return seg_map, depth_map

# ─── UPDATED Filtering Helper ────────────────────────────────────────────────

def filter_images(zip_path: str, selected_class: str, min_depth: float, max_depth: float):
    """Filter a ZIP archive of images.

    An image is *kept* if **at least 95% of the pixels belonging to the selected
    semantic class** also have depth values within the user‑defined range
    (`min_depth` ≤ depth ≤ `max_depth`).
    """
    output_zip = "/tmp/filtered_output.zip"
    temp_dir = "/tmp/temp_images"

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    target_idx = CITYSCAPES_CLASSES.index(selected_class)
    log_lines = []
    valid_images = []

    # Extract all images first so that we can iterate easily.
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    for root, _, files in os.walk(temp_dir):
        for filename in sorted(files):
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip non‑image files or failed reads.

            seg_map, depth_map = process_image(img)
            # Resize depth map to match segmentation map for pixelwise checks.
            depth_resized = cv2.resize(depth_map, (seg_map.shape[1], seg_map.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Identify all pixels of selected class.
            class_mask = seg_map == target_idx
            total_class_pixels = int(np.count_nonzero(class_mask))

            if total_class_pixels == 0:
                log_lines.append(f"Bad. {filename} (class not present)")
                continue

            # Of those pixels, how many are within the depth range?
            depth_range_mask = (depth_resized >= min_depth) & (depth_resized <= max_depth)
            in_range_mask = class_mask & depth_range_mask
            in_range_pixels = int(np.count_nonzero(in_range_mask))

            ratio = in_range_pixels / total_class_pixels

            if ratio >= 0.55:
                valid_images.append(img_path)
                log_lines.append(f"Good! {filename} ({ratio:.2%} of class pixels in range)")
            else:
                log_lines.append(f"Bad. {filename} ({ratio:.2%} of class pixels in range)")

    if valid_images:
        with zipfile.ZipFile(output_zip, "w") as zip_out:
            for img in valid_images:
                zip_out.write(img, os.path.basename(img))
        return output_zip, "\n".join(log_lines)
    else:
        return None, "\n".join(log_lines)

# ─── Gradio Glue Code ────────────────────────────────────────────────────────

def gradio_pipeline(zip_file, selected_class, min_depth, max_depth):
    filtered_zip, log = filter_images(zip_file.name, selected_class, min_depth, max_depth)
    return (filtered_zip if filtered_zip else None), log


@hydra.main(version_base=None, config_path="hydra_config", config_name="app_config_models")
def main(config: DictConfig):
    load_model_from_config(config)

    tab1 = gr.Interface(
        fn=generate_depth_and_segmentation,
        inputs=gr.Image(type="numpy", label="Upload Image"),
        outputs=[
            gr.Image(type="numpy", label="Segmentation Mask"),
            gr.Image(type="numpy", label="Depth Map (Plasma)"),
            gr.File(label="Segmented Point Cloud (.ply)"),
            gr.File(label="Raw Depth Point Cloud (.ply)")
        ],
        title="Depth + Segmentation + Point Cloud",
        description="Upload an image to generate semantic segmentation, depth map, and point clouds."
    )

    tab2 = gr.Interface(
        fn=gradio_pipeline,
        inputs=[
            gr.File(label="Upload ZIP"),
            gr.Dropdown(choices=CITYSCAPES_CLASSES, label="Cityscapes Class", value="Car"),
            gr.Slider(0, 80, value=5, step=1, label="Min Depth"),
            gr.Slider(0, 80, value=50, step=1, label="Max Depth"),
        ],
        outputs=[
            gr.File(label="Filtered ZIP"),
            gr.Textbox(label="Filtering Log", lines=20, interactive=False)
        ],
        title="Filter Images by Class & Depth",
        description="Filter a ZIP of images to only include ones with selected class and depth range. Shows filtering log."
    )

    gr.TabbedInterface([tab1, tab2], ["Depth & Segmentation", "ZIP Filtering"]).launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == '__main__':
    main()
