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
import hydra
from omegaconf import DictConfig

from depth_anything_v2.dpt import DepthSegmentAnythingJointV2
from transform import PrepareForNet, Resize, NormalizeImage

@hydra.main(version_base=None, config_path="hydra_config", config_name="app_config_models")
def main(config: DictConfig):
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    }
    joint_model = DepthSegmentAnythingJointV2(**{**model_configs['vitb'], 'max_depth': 80})
    state_metric_depth_seg = config.app.ckptpath  # Replace with actual path
    state_dict = torch.load(state_metric_depth_seg)
    joint_model.load_state_dict(state_dict['state_dict'], strict=False)
    joint_model.cuda() if torch.cuda.is_available() else joint_model.cpu()
    joint_model.eval()

    CITYSCAPES_CLASSES = [
        "Road",
        "Sidewalk", 
        "Building", 
        "Wall", 
        "Fence",
        "Pole", 
        "Traffic Light", 
        "Traffic Sign", 
        "Vegetation", 
        "Terrain", 
        "Sky", 
        "Person", 
        "Rider", 
        "Car", 
        "Truck", 
        "Bus", 
        "Train", 
        "Motorcycle", 
        "Bicycle"
    ]

    CITYSCAPES_COLORS = [
        [128, 64, 128], 
        [244, 35, 232], 
        [70, 70, 70], 
        [102, 102, 156],
        [190, 153, 153], 
        [153, 153, 153], 
        [250, 170, 30], 
        [220, 220, 0],
        [107, 142, 35], 
        [152, 251, 152], 
        [0, 130, 180], 
        [220, 20, 60],
        [255, 0, 0], 
        [0, 0, 142], 
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100], 
        [0, 0, 230], 
        [119, 11, 32]
    ]

    def image2tensor(raw_image, input_size=518):

        '''
        Function used for preprocessing an image at before being fed into de network.
        '''

        transform = Compose([
            Resize(width=input_size, height=input_size, resize_target=False,
                keep_aspect_ratio=True, ensure_multiple_of=14,
                resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        image = raw_image / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        return image.to('cuda' if torch.cuda.is_available() else 'cpu'), raw_image.shape[:2]

    def apply_color_map(segmentation, colormap):
        '''
        Segmentation map for visualization purposes, mapping the 0-19 to a colorspace.
        '''
        h, w = segmentation.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in enumerate(colormap):
            colored[segmentation == label] = color
        return colored

    def apply_colormap_to_depth(depth, cmap_name='plasma'):
        '''
        Depth map for visualization purposes, we nomralize the depth values and we use a plasma cmap name.
        '''
        norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-6)
        colormap = cm.get_cmap(cmap_name)
        colored = colormap(norm)[:, :, :3]
        return (colored * 255).astype(np.uint8)

    def create_point_cloud(depth_map, rgb_colors, save_path):
        '''
        Function that creates the point cloud from an depth map and some rgb colors projecting the 2d points,
        into a 3d space with f_x, f_y, c_x, c_y.
        '''
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
        points, colors = points[valid], colors[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(save_path, pcd)
        return save_path

    def generate_depth_and_segmentation(image):
        '''
        Function that makes a forward pass of an image through the architecture, and returns the segmenation mapped to a colorspace,
        depth to a plasma cmap normalized values, and the raw and segemented pointcloud of the image.
        '''
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        segmentation, depth = joint_model.infer_image(image_bgr)

        segmentation = torch.argmax(segmentation, dim=1).float()  # No squeeze(0) here
        segmentation = F.interpolate(segmentation.unsqueeze(1), (1024, 2048), mode="nearest")[0, 0]  # Use nearest neighbor
        segmentation = segmentation.cpu().numpy().astype(np.uint8)
        colored_seg = apply_color_map(segmentation, CITYSCAPES_COLORS)

        depth_np = depth.squeeze()
        depth_colored = apply_colormap_to_depth(depth_np)

        rgb_resized = cv2.resize(image, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        seg_pcd = create_point_cloud(depth_np, colored_seg, "/tmp/seg_pointcloud.ply")
        raw_pcd = create_point_cloud(depth_np, rgb_resized, "/tmp/raw_pointcloud.ply")

        return colored_seg / 255., depth_colored / 255., seg_pcd, raw_pcd

    def process_image(image):
        '''
        Function used to only to a forward pass through the model.
        Used for the zip inference.
        '''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        segmentation, depth = joint_model.infer_image(image)
        segmentation = torch.argmax(segmentation, dim=1).squeeze(0).float()
        segmentation = F.interpolate(segmentation.unsqueeze(0).unsqueeze(0), (1024, 2048), mode="bilinear", align_corners=True)[0, 0]
        return segmentation.cpu().numpy().astype(np.uint8), depth.squeeze()

    def filter_images(zip_path, selected_class, min_depth, max_depth):
        '''
        Function that receives a zip file containing images
        '''
        output_zip = "/tmp/filtered_output.zip"
        temp_dir = "/tmp/temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        target_class = CITYSCAPES_CLASSES.index(selected_class)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        valid_images = []
        for root, _, files in os.walk(temp_dir):
            for filename in files:
                img_path = os.path.join(root, filename)
                print(img_path)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                seg, depth = process_image(image)
                mask = (seg == target_class) & (depth >= min_depth) & (depth <= max_depth)

                if np.count_nonzero(mask) >= 100:
                    valid_images.append(img_path)

        if valid_images:
            with zipfile.ZipFile(output_zip, 'w') as zip_out:
                for img in valid_images:
                    zip_out.write(img, os.path.basename(img))
            return output_zip
        else:
            return None

    def gradio_pipeline(zip_file, selected_class, min_depth, max_depth):
        output_zip = filter_images(zip_file.name, selected_class, min_depth, max_depth)
        return output_zip if output_zip else "No valid images found."

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
        outputs=gr.File(label="Filtered ZIP"),
        title="Filter Images by Class & Depth",
        description="Filter a ZIP of images to only include ones with selected class and depth range."
    )

    gr.TabbedInterface([tab1, tab2], ["Depth & Segmentation", "ZIP Image Filter"]).launch(
        server_name="0.0.0.0", server_port=7860, share=False
    )
