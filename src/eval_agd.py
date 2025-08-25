import os
import sys

import wandb
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import copy

import torch
import torch.nn.functional as F

from utils.file_utils import load_config
from inference import AffordanceInference
from utils.vlm_utils import get_text_embedding_options
from PIL import Image

from utils.eval_utils import (
    cal_kl, cal_sim, cal_nss
)
from utils.img_utils import grid_visualize, transform_imgs

import argparse
from tqdm import tqdm

### Util functions
def output_upsample(out, target_size, mode):
    """
    upsample output to target size
    """
    out = torch.tensor(out).unsqueeze(0).unsqueeze(0)
    if mode == "repeat":
        return F.interpolate(out, size=target_size, mode="nearest").squeeze().numpy()
    elif mode == "bilinear":
        return F.interpolate(out, size=target_size, mode="bilinear", align_corners=False, antialias=True).squeeze().numpy()
    else:
        raise NotImplementedError


def resize_to_multiple_of_14(img, max_size=672):
    """
    Resize image so that h and w are multiples of 14 (required for inference)
    Optionally limit the maximum size of the longer side
    
    Args:
        img: Input image as numpy array
        max_size: Maximum length of the longer side (optional)
    
    Returns:
        resized image and original shape for later restoration
    """
    original_h, original_w = img.shape[:2]
    
    # Start with original dimensions
    target_h, target_w = original_h, original_w
    
    # If max_size is specified, scale down if necessary
    if max_size is not None:
        max_dim = max(original_h, original_w)
        if max_dim > max_size:
            scale = max_size / max_dim
            target_h = int(original_h * scale)
            target_w = int(original_w * scale)
    
    # Find nearest dimensions that are multiples of 14
    new_h = ((target_h + 13) // 14) * 14  # Round up to nearest multiple of 14
    new_w = ((target_w + 13) // 14) * 14  # Round up to nearest multiple of 14
    
    # Resize image
    if len(img.shape) == 3:
        resized_img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    else:
        # Handle grayscale
        resized_img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    
    return resized_img, (original_h, original_w)
    

### Eval on AGD20K
def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--agd_root", required=True, help="Path to AGD20K root")
    parser.add_argument("--viz_dir", required=False, default=None, help="Path to save visualization")
    
    args = parser.parse_args()
    
    ### Load model
    # Load config for additional settings
    cfg = load_config(args.config)

    # build inference
    text_embedding_option = "embeddings_oai"
    print(f"Using text embedding option: {text_embedding_option}")
    text_embedding_func = get_text_embedding_options(text_embedding_option)
    inference = AffordanceInference(args.config, args.checkpoint, text_embedding_func)

    ### Load eval data
    if args.viz_dir is not None:
        os.makedirs(args.viz_dir, exist_ok=True)

    agd_egocentric_dir = os.path.join(args.agd_root, "egocentric")
    agd_gt_dir = os.path.join(args.agd_root, "GT")

    # build all info
    eval_set_info = []
    # Rephrase to disambiguate for specific action-object combinations
    disambiguation_set = {
        ("hit", "axe"): "handle of axe to hold during hitting",
        ("ride", "bicycle"): "region to sit on and push the bicycle",
        ("pour", "cup"): "handle of the cup to hold while pouring",
        ("wash", "cup"): "rim of the cup to wash",
        ("hold", "cup"): "handle to hold the cup"
    }
    
    for action_name in os.listdir(agd_gt_dir):
        action_dir = os.path.join(agd_gt_dir, action_name)
        for obj_name in os.listdir(action_dir):
            obj_dir = os.path.join(action_dir, obj_name)
            for i in range(len(os.listdir(obj_dir))):
                img_name = os.listdir(obj_dir)[i]
                gt_path = os.path.join(agd_gt_dir, action_name, obj_name, img_name)
                img_path = os.path.join(agd_egocentric_dir, action_name, obj_name, img_name.replace(".png", ".jpg"))
                
                if (action_name, obj_name) in disambiguation_set:
                    text_desc = disambiguation_set[(action_name, obj_name)]
                else:
                    text_desc = f"region to {action_name} the {obj_name}"

                eval_set_info.append({
                    "img_path": img_path,
                    "gt_path": gt_path,
                    "text_desc": text_desc,
                    "viz_name": f"{action_name}_{obj_name}_{i}" # for visualization
                })

    KLs = []
    SIMs = []
    NSSs_01 = [] # original NSS 
    NSSs_05 = [] # NSS with threshold 0.5 (see Appendix B for details)
    
    idx = 0
    for data in tqdm(eval_set_info):
        # load eval data
        img = np.array(Image.open(data["img_path"]).convert("RGB"))
        gt_mask = plt.imread(data["gt_path"])
        text = data["text_desc"]

        if len(img.shape) == 2: # handle grayscale images
            img = np.stack([img, img, img], axis=-1)

        # Resize image to satisfy multiple-of-14 constraint for inference
        img_resized, original_shape = resize_to_multiple_of_14(img, max_size=672)

        # inference on resized image
        out = inference.predict(img_resized, text, thresh=None)
        
        # Resize output back to original image dimensions
        out = output_upsample(out, original_shape, mode="bilinear")

        # postprocess
        out = np.clip(out, 1e-3, 1-(1e-3))

        # save visualization of output
        grid_visualize(
            img_list=[img, out, gt_mask],
            name_list=["orig img", "model pred", "gt"],
            save_path=f"{args.viz_dir}/{data['viz_name']}.png",
            n_rows=1,
            title=f"{data['viz_name']}"
        )

        # compute metrics
        kld, sim, nss_01, nss_05 = cal_kl(out, gt_mask), cal_sim(out, gt_mask), cal_nss(out, gt_mask, threshold=0.1), cal_nss(out, gt_mask, threshold=0.5)
        KLs.append(kld)
        SIMs.append(sim)
        NSSs_01.append(nss_01)
        NSSs_05.append(nss_05)

    print(f"KL: {np.mean(KLs)}, SIM: {np.mean(SIMs)}, NSS: {np.mean(NSSs_01)}, NSS_05: {np.mean(NSSs_05)}")

if __name__ == "__main__":
    eval()