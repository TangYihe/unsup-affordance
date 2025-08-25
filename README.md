## UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation

#### [[Project Page]](https://unsup-affordance.github.io/) [[Paper]](https://unsup-affordance.github.io/) 

[Yihe Tang](https://tangyihe.com/)<sup>1</sup>, [Wenlong Huang](https://wenlong.page)<sup>1</sup>, [Yingke Wang](https://www.wykac.com/)<sup>1</sup>, [Chengshu Li](https://www.chengshuli.me/)<sup>1</sup>, [Roy Yuan](https://www.linkedin.com/in/ryuan19)<sup>1</sup>, [Ruohan Zhang](https://ai.stanford.edu/~zharu/)<sup>1</sup>, [Jiajun Wu](https://jiajunwu.com/)<sup>1</sup>, [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)<sup>1</sup>

<sup>1</sup>Stanford University

### Overview

This is the official codebase for [UAD](https://unsup-affordance.github.io/). UAD is a method that distills affordance knowledge from foundation models into a task-conditioned affordance model without any manual annotations.

This repo contains:
- [UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation](#uad-unsupervised-affordance-distillation-for-generalization-in-robotic-manipulation)
    - [\[Project Page\] \[Paper\]](#project-page-paper)
  - [Overview](#overview)
  - [Environment Setup](#environment-setup)
  - [Affordance Model Training and Inference](#affordance-model-training-and-inference)
  - [Object Rendering Pipeline](#object-rendering-pipeline)
    - [B1K assets](#b1k-assets)
    - [Objaverse assets](#objaverse-assets)
  - [Dataset Curation Pipeline](#dataset-curation-pipeline)
  - [AGD20K Evaluation](#agd20k-evaluation)


### Environment Setup
- (Optioal) If you are using [Omnigibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html) for rendering [Behavior1K](https://behavior.stanford.edu/behavior-1k) assets, or Blender for rendering [Objaverse](https://github.com/allenai/objaverse-xl) assets, please follow their installation guide respectively. 
  - Note the rendering libraries may have version conflicts with the data pipeline / model training code, consider using a separate env in that case. 
- (Optional) If you are using open-sourced [sentence-transformers](https://github.com/UKPLab/sentence-transformers?tab=readme-ov-file#installation) for language embedding, please follow their installation guide. 
  - We recommend installing from source
- Create your conda environment and install torch
  ```
  conda create -n uad python=3.9
  conda activate uad
  pip install torch torchvision torchaudio
  ```
- Install unsup-affordance in the same conda env
  ```
  git clone https://github.com/TangYihe/unsup-affordance.git
  cd unsup-affordance
  pip install -r requirements.txt
  pip install -e .
  ```

### Affordance Model Training and Inference

We provide options to embed language with OpenAI api, or open-sourced [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

- To run inference with our trained checkpoints, run:
   ```
   # use sentence-transformers embedding
   python src/inference.py --config configs/st_emb.yaml --checkpoint checkpoints/st_emb.pth

   # use openai embedding (make sure you've properly set OPENAI_API_KEY env variable)
   python src/inference.py --config configs/oai_emb.yaml --checkpoint checkpoints/oai_emb.pth
   ```
   The script will run "twist open" query on ```examples/example_image.png``` and save output to ```examples/affordance_map.png```.

- To run training on our provided or your own dataset: 
  Our provided dataset could be found in the [Google Drive](https://drive.google.com/drive/folders/1FKrwGdCDgbIUaRtcrR0SAyNENFVnzIa-?usp=sharing)
  1. Create the torch dataset from h5 files by running
     ```
     python src/model/dataset.py --data_root YOUR_DATA_DIR 
     ```
     This will save a .pt dataset under ```YOUR_DATA_DIR/dataset/```  
     Arguments:
     - only process certain categories (by default all): ```--categories CATEGORY1 CATEGORY2```
     - choose embedding type (by default oai embedding): ```--embedding_type EMBEDDING_TYPE```
  
  2. Train with your saved dataset by running
     ```
     python src/train.py --config YOUR_CONFIG_YAML --data YOUR_DATASET_PT --run_name YOUR_RUN_NAME
     ```
     The logs will be saved under ```logs/yrmtdt/YOUR_RUN_NAME/ckpts```  
     We found using some image to replace the white background of the renderings would improve model training. In our experiments, we used indoor renderings from [Behavior Vision Suite](https://behavior-vision-suite.github.io), which could be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1FKrwGdCDgbIUaRtcrR0SAyNENFVnzIa-?usp=sharing). To enable background augmentation, please set the directory of your image folder to ```dataset_bg_dir``` in your config file. 
     Arguments:
     - multiple dataset: ```--data DATASET_1_PATH DATASET_2_PATH ```
     - set batch size / lr / epochs: ```--lr LR --batch BATCH_SIZE --epochs NUM_EPOCHS```
     - resume training: ```--resume_ckpt CKPT_PATH```
     - turn off wandb logging: ```--no_wandb```


### Object Rendering Pipeline
We provide code to render Behavior-1K assets with Omnigibson, or Objaverse assets with Blender. 

#### B1K assets
The code is in behavior1k_omnigibson_render. 
- Unzip ```qa_merged.zip```
- Render assets:
  ```
  python render.py --orientation_root ORI_ROOT --og_dataset_root OG_DATASET_ROOT --category_model_list selected_object_models.json --save_path YOUR_DATA_DIR
  ```
  Note: ORI_ROOT is the folder of your unzipped ```qa_merged/```. OG_DATASET_ROOT is your Omnigibson objects path, shall be ```YOUR_OG_PATH/omnigibson/data/og_dataset/objects```.
- Convert the renderings to .h5 format:
  ```
  python convert_b1k_data_with_crop.py --data_root YOUR_DATA_DIR
  ```

#### Objaverse assets
The code is in objaverse_blender_render. 
- Download the objaverse assets, run
  ```
  python objaverse_download_script.py --data_root YOUR_DATA_DIR --n N
  ```
  - N is the number of assets you want to download from each category. By default 50.
  - In our case study, we have used a subset from the lvis categories. You can change the category used in the script. 
- Filter out assets with transparent (no valid depth) or too simple texture, run
   ```
   python texture_filter.py --data_root YOUR_DATA_DIR
   ```
- Render the assets with Blender 
   ```
   blender --background \
   --python blender_script.py -- \
   --data_root YOUR_DATA_DIR \
   --engine BLENDER_EEVEE_NEXT \
   --num_renders 8 \
   --only_northern_hemisphere
   ```
- Convert the renderings to .h5 format
  ```
  python h5_conversion.py --data_root=YOUR_DATA_DIR
  ```


### Dataset Curation Pipeline
Pipeline to perform DINOv2 feature 3D fusion, clustering, VLM proposal and computing affordance maps. The current implementation uses gpt-4o, so requires properly setting ```OPENAI_API_KEY``` env variable. 

```
python pipeline.py --base_dir=YOUR_DATA_DIR --embedding_type=YOUR_EMBEDDING_TYPE
```
Arguments: 
- ```--use_data_link_segs```: pass in when using Behavior-1K data
- ```--top_k K```: use the best K views of render in the final dataset for training (default is 3)
- ```--category_names CATEGORY1 CATEGORY2```: only process certain categories

### AGD20K Evaluation
To evaluate our trained model on AGD20K Unseen testset, run
```
python src/eval_agd.py --config configs/eval_agd.yaml --checkpoint checkpoints/eval_agd.pth --agd_root YOUR_AGD_TESTSET_DIR 
```
Arguments:
- ```--agd_root``` is the path to the Unseen testset. It shall be the parent directory for ```egocentric``` and ```GT```. 
- ```--viz_dir``` Optional. Pass in to save visualizations of predictions. 

Notes:
- We additionally report the metric NSS-0.5, which is computed by changing the ground truth binarization threshold to 0.5. Please see Appendix B for details. 
- This model requires OpenAI language embedding. 

