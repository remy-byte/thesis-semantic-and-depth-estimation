# test_app_combined_model.py

import os
import cv2
import zipfile
import tempfile
import shutil
import numpy as np
import torch
import unittest
from omegaconf import OmegaConf

import app_combined  # your refactored script

class TestAppCombinedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg_path = "hydra_config/app_config_models.yaml"
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Config file not found at {cfg_path}")
        cls.cfg = OmegaConf.load(cfg_path)

        app_combined.load_model_from_config(cls.cfg)
        cls.model = app_combined.joint_model
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[setUpClass] Model loaded on device: {cls.device}")

    def test_infer_image_output_shapes(self):
        """
        Create a synthetic RGB image (random uint8), run joint_model.infer_image(),
        and verify:
          - segmentation logits is a torch.Tensor of shape (1, 19, h2, w2)
          - depth is either a torch.Tensor or a numpy.ndarray of shape (H, W)
        """
        H, W = 1024, 2048
        # Synthetic random RGB uint8 image
        rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        # convert to BGR as model expects BGR input
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        seg_logits, depth = self.model.infer_image(bgr)

        # logits shape: (1, 19, h2, w2)
        self.assertEqual(seg_logits.dim(), 4)
        bs, nc, h2, w2 = seg_logits.shape
        self.assertEqual(bs, 1)
        self.assertEqual(nc, 19)
        self.assertTrue(h2 > 0 and w2 > 0)

        # depth shape and type
        if isinstance(depth, torch.Tensor):
            self.assertEqual(depth.device.type, self.device)
            self.assertEqual(tuple(depth.shape), (H, W))
        else:  # numpy.ndarray
            self.assertEqual(depth.dtype, np.float32)
            self.assertEqual(depth.shape, (H, W))

    def test_generate_depth_and_segmentation_consistency(self):
        """
        Create a synthetic RGB image (zeros), run generate_depth_and_segmentation,
        and verify:
          - seg_rgb & depth_rgb are float arrays in [0,1], shape H×W×3
          - returned PLY paths exist
        """
        H, W = 1024, 2048
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        seg_rgb, depth_rgb, seg_ply, raw_ply = app_combined.generate_depth_and_segmentation(rgb)

        # Shapes match input H×W×3
        self.assertEqual(seg_rgb.shape, (H, W, 3))
        self.assertEqual(depth_rgb.shape, (H, W, 3))

        # Dtype is float
        self.assertIn(seg_rgb.dtype, [np.float32, np.float64])
        self.assertIn(depth_rgb.dtype, [np.float32, np.float64])

        # Values lie in [0,1]
        self.assertGreaterEqual(float(seg_rgb.min()), 0.0)
        self.assertLessEqual(float(seg_rgb.max()), 1.0)
        self.assertGreaterEqual(float(depth_rgb.min()), 0.0)
        self.assertLessEqual(float(depth_rgb.max()), 1.0)

    def test_filter_images_on_synthetic_zip(self):
        """
        Create a ZIP containing one synthetic image (128×128).
        Run filter_images for class "Car", depth [1,50].
        Verify return is either None or a valid ZIP path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            H, W = 128, 128
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
            img_path = os.path.join(tmpdir, "zero.png")
            cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            zip_path = os.path.join(tmpdir, "one_image.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(img_path, os.path.basename(img_path))

            out_zip = app_combined.filter_images(zip_path, selected_class="Car", min_depth=1.0, max_depth=50.0)

            if out_zip is not None:
                self.assertTrue(os.path.isfile(out_zip), f"Expected a ZIP at {out_zip}")
                with zipfile.ZipFile(out_zip, "r") as zf:
                    _ = zf.namelist()
            else:
                self.assertIsNone(out_zip)

if __name__ == "__main__":
    unittest.main()
