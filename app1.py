import gradio as gr
import os
import cv2
import numpy as np
import torch
import open3d as o3d
from torchvision.transforms import Compose
from depth_anything_v2.dpt import DepthSegmentAnythingV2
import torch.nn.functional as F
from transform import PrepareForNet, Resize, NormalizeImage
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="hydra_config", config_name="app_config_models")
def main(config: DictConfig):
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    }
    model = DepthSegmentAnythingV2(**{**model_configs['vitb'], 'max_depth': 80})
    state_metric_depth_seg = config.app.ckptpath  # Replace with actual path
    state_dict = torch.load(state_metric_depth_seg)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # Function to apply color map based on segmentation labels
    def apply_color_map(segmentation, colormap):
        h, w = segmentation.shape
        colored_segmentation = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in enumerate(colormap):
            colored_segmentation[segmentation == label] = color
        return colored_segmentation

    colors = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]

    def image2tensor(raw_image, input_size=518):
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
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        return image.to(DEVICE), raw_image.shape[:2]

    def create_point_cloud_from_depth(depth_map, colored_segmentation ,focal_length=(2262.52, 2265.30), center=(1024, 512), save_path="/home/bue6clj/Depth-Anything-V2/metric_depth/point_cloud_app.ply"):
        height, width = depth_map.shape
        f_x, f_y = focal_length
        c_x, c_y = center
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - c_x) / f_x
        y = (y - c_y) / f_y
        z = depth_map

        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

        colors = np.array(colored_segmentation).reshape(-1, 3) / 255.0  # Normalize to [0, 1]

        points = points[z.reshape(-1) > 0]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)


        # Save point cloud to file
        o3d.io.write_point_cloud(save_path, pcd)
        
        return save_path

    def generate_depth_and_segmentation(image):
        img_tensor, (h, w) = image2tensor(image, 518)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        segmentation, depth = model.infer_image(image)
        segmentation = torch.argmax(segmentation, dim=1).squeeze(0).float()
        segmentation = F.interpolate(segmentation.unsqueeze(0).unsqueeze(0),
                                    (1024, 2048), mode="bilinear", align_corners=True)[0, 0]
        segmentation = segmentation.cpu().detach().numpy().astype(np.uint8)

        colored_segmentation = apply_color_map(segmentation, colors)


        point_cloud_path = create_point_cloud_from_depth(depth, colored_segmentation=colored_segmentation)
        return colored_segmentation / 255., depth / 255., point_cloud_path

    def visualize_image(image):
        segmentation_mask, depth_map, point_cloud_path = generate_depth_and_segmentation(image)
        return segmentation_mask, depth_map, point_cloud_path

    demo = gr.Interface(
        fn=visualize_image,
        inputs=gr.Image(type="numpy", label="Upload Image"),
        outputs=[
            gr.Image(type="numpy", label="Segmentation Mask"),
            gr.Image(type="numpy", label="Depth Map"),
            gr.Model3D(clear_color=[1.0, 1.0, 1.0, 1.0], label="3D Point Cloud Visualization")
        ],
        title="Segmentation, Depth, and Point Cloud Visualization",
        description="Upload an image to generate segmentation, depth map, and 3D point cloud visualization.",
        live=True
    )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == '__main__':
    main()