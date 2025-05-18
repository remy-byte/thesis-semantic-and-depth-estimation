import gradio as gr
import hydra
from omegaconf import DictConfig
import zipfile
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from torchvision.transforms import Compose
from depth_anything_v2.dpt import DepthSegmentAnythingV2
from transform import PrepareForNet, Resize, NormalizeImage

@hydra.main(version_base=None, config_path="hydra_config", config_name="app_config_models")
def main(config: DictConfig):
    # Cityscapes 19 classes
    CITYSCAPES_CLASSES = [
        "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", 
        "Traffic Light", "Traffic Sign", "Vegetation", "Terrain", 
        "Sky", "Person", "Rider", "Car", "Truck", "Bus", 
        "Train", "Motorcycle", "Bicycle"
    ]

    # Image Preprocessing
    def image2tensor(raw_image, input_size=518):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        image = raw_image / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        return image.to(DEVICE)

    # Load Model
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
    }
    model = DepthSegmentAnythingV2(**{**model_configs['vitb'], 'max_depth': 80})
    state_dict = torch.load('..')
    model.load_state_dict(state_dict['state_dict'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # Process image
    def process_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        segmentation, depth = model.infer_image(image)
        segmentation = torch.argmax(segmentation, dim=1).squeeze(0).float()
        segmentation = segmentation.unsqueeze(0).unsqueeze(0)
        segmentation = F.interpolate(segmentation, (1024, 2048), mode="bilinear", align_corners=True)[0, 0]
        segmentation = segmentation.squeeze(0).squeeze(0)
        # Simulate a point cloud from the depth map (same as before)
        segmentation = segmentation.cpu().detach().numpy()
        segmentation = segmentation.astype(np.uint8)
        return segmentation, depth

    # Process ZIP file
    def filter_images(zip_path, selected_class, min_depth, max_depth):
        output_zip = "filtered_output.zip"
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        target_class = CITYSCAPES_CLASSES.index(selected_class)  # Convert class name to index

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        valid_images = []
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            for img in os.listdir(filepath):
                to_read = os.path.join(filepath,img)
                print(to_read)
                image = cv2.imread(to_read)
                if image is None:
                    print('aici')
                    continue
            
                segmentation, depth = process_image(image)
                print(np.unique(segmentation))
                mask = (segmentation == target_class) & (depth >= min_depth) & (depth <= max_depth)
                if np.any(mask):
                    valid_images.append(to_read)
        
        if valid_images:
            with zipfile.ZipFile(output_zip, 'w') as zip_out:
                for valid_image in valid_images:
                    zip_out.write(valid_image, os.path.basename(valid_image))

            return output_zip if valid_images else None

    # Gradio UI
    def gradio_pipeline(zip_file, selected_class, min_depth, max_depth):
        output_zip = filter_images(zip_file.name, selected_class, min_depth, max_depth)
        return output_zip if output_zip else "No valid images found."

    gr.Interface(
        fn=gradio_pipeline,
        inputs=[
            gr.File(label="Upload ZIP File"),
            gr.Dropdown(choices=CITYSCAPES_CLASSES, label="Select Cityscapes Class", value="Car"),
            gr.Slider(0, 100, value=5, step=1, label="Min Depth"),
            gr.Slider(0, 100, value=50, step=1, label="Max Depth"),
        ],
        outputs=gr.File(label="Filtered ZIP Output"),
        title="Cityscapes Image Filtering",
        description="Upload a ZIP file of images. The model filters images containing the selected Cityscapes class within the depth range.",
    ).launch(
        server_name="0.0.0.0",  # Allow remote access
        server_port=7861,       # Use the port you specified
        share=False              # Set to True if you need a public link
    )

if __name__ == '__main__':
    main()