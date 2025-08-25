#!/usr/bin/env python3

import argparse
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from model.network import Conv2DFiLMNet

from utils.img_utils import transform_imgs, load_pretrained_dino, get_dino_features_from_transformed_imgs
from utils.vlm_utils import get_text_embedding_options
from utils.file_utils import load_config


class AffordanceInference:
    def __init__(self, config_path, checkpoint_path, text_embedding_func):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        cfg = load_config(config_path)
        model_cfg = cfg["model"]
        
        # Build model
        self.model = Conv2DFiLMNet(**model_cfg)
        self.model.build()
        self.model.to(self.device)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        # Load DINO
        torch_home = cfg.get("torch_home", None)
        dino_model_type = cfg.get("dino_model_type", "dinov2_vits14")
        dino_use_registers = cfg.get("dino_use_registers", True)
        self.dino = load_pretrained_dino(dino_model_type, use_registers=dino_use_registers, torch_path=torch_home).to(self.device).eval()

        self.text_embedding_func = text_embedding_func
    
    @torch.no_grad()
    def predict(self, img_np, text, thresh=0.5):
        """
        img_np: H×W×3 numpy array (uint8 or float in [0,1])
        text: natural language description
        thresh: threshold for binary output
        Returns: similarity map as numpy array
        """
        # Preprocess image
        proc = transform_imgs(img_np, blur=False)[0]
        proc = proc.unsqueeze(0).to(self.device)
        
        # Get text embedding
        lang_emb = torch.from_numpy(self.text_embedding_func(text))
        lang_emb = lang_emb.to(self.device).unsqueeze(0).to(torch.float32)
        
        # Get DINO features
        feat = get_dino_features_from_transformed_imgs(self.dino, proc, repeat_to_orig_size=False)
        feat = feat.permute(0, 3, 1, 2)
        
        # Forward pass
        logits = self.model(feat, lang_emb).squeeze(0).squeeze(0)
        sim = torch.sigmoid(logits)
        
        if thresh is not None:
            sim = (sim > thresh).float()
        
        sim_np = sim.cpu().numpy()
        
        # Resize to original image size
        H, W = img_np.shape[:2]
        sim_np = np.array(
            T.functional.resize(
                Image.fromarray(sim_np.astype(np.float32), mode="F"),
                (H, W),
                interpolation=T.InterpolationMode.BILINEAR
            )
        )
        
        return sim_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    # Load config for additional settings
    cfg = load_config(args.config)
    
    ## ===== Example usage ===== ##
    image_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'example_image.png')
    text_query = "twist open"
    
    # Load image
    img = np.array(Image.open(image_path).convert("RGB"))
    
    # Get text embedding function
    text_embedding_option = cfg.get("text_embedding", "embeddings_oai")
    print(f"Using text embedding option: {text_embedding_option}")
    text_embedding_func = get_text_embedding_options(text_embedding_option)

    # Initialize inference
    inference = AffordanceInference(args.config, args.checkpoint, text_embedding_func)
    
    # Post-processing threshold for shutting down low affordance regions
    shutdown_thresh = cfg.get("thresh", 0.5)
    
    # Run prediction
    result = inference.predict(img, text_query, shutdown_thresh)
    
    print(f"Predicted affordance map for '{text_query}' with shape {result.shape}")
    
    # Display result
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='hot')
    plt.title(f"Affordance Map: '{text_query}'")
    plt.axis('off')
    
    plt.tight_layout()
    
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'examples', 'affordance_map.png'))


if __name__ == "__main__":
    main()