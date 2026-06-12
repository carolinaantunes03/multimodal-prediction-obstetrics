import sys
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("../multimodal_architecture"))
sys.path.append(os.path.abspath("../embeddings-optimisation/edca/edca"))
import data as data
from data import get_val_image_transform
from PIL import Image
import models as models

import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

retrospective_dataset = "../datasets/retrospective-dataset/all_retrospective_data.csv"
prospective_dataset = "../datasets/prospective-dataset/all_prospective_data.csv"

with open("multimodal_architecture/columns_config.json", "r") as f:
    col_config = json.load(f)
num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
image_cols = col_config["image_cols"]

processo = "Processo"
label = "Class"

multimodal_model_path = "../multimodal_architecture/models_prospective/MedViT2_nopt/final_model.pth"


pipeline_path = "utils/final_pipeline_edca.pkl"
pipeline = joblib.load(pipeline_path)


train_df = pd.read_csv (retrospective_dataset)
train_df = train_df.drop(columns=[processo, label], errors="ignore")



def load_multimodal_model (multimodal_model_path, num_cols, cat_cols): 
    
    num_numerical = len(num_cols)
    num_categories = [train_df[col].nunique() + 1 for col in cat_cols]


    multimodal_model = models.build_multimodal_model(
        num_numerical=num_numerical,
        num_categories=num_categories,
        tabular_token_dim=192,
        tabular_hidden_dim=192
    ).to(DEVICE)

    multimodal_model.load_state_dict(torch.load(multimodal_model_path, map_location=DEVICE))
    multimodal_model.eval()

    return multimodal_model

image_transform = get_val_image_transform()

def load_image(img_path, root_dir=None):
  
    if not isinstance(img_path, str) or pd.isna(img_path):
        return torch.zeros(3, 224, 224), 0
        
    
    if root_dir and not os.path.isabs(img_path):
        img_path = os.path.join(root_dir, img_path)
    
    if not os.path.exists(img_path):
        print(f"[WARN] Missing image path: {img_path}")
        return torch.zeros(3, 224, 224), 0
    
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Could not open {img_path}: {e}")
        return torch.zeros(3, 224, 224), 0
    
  
    return image_transform(img),1


@torch.no_grad()
def extract_image_embeddings (model, image, valid):

    model.eval()

 
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    emb,_ = model.image_encoder(
            image,
            image_valid_num=torch.tensor([1]).to(DEVICE),
            return_feature = True
            )  
    
    emb = F.layer_norm (emb, emb.shape[1:]) #the embedding is normalized (to stabilize values)
    
    return emb.cpu().numpy().flatten()   #flattened into a 1D vector
   

def get_tabular_features (data, num_cols, cat_cols):

    features = []

    cols_to_drop = []
    if "Processo" in data.index:
        cols_to_drop.append("Processo")
    if "Class" in data.index:
        cols_to_drop.append("Class")

    data = data.drop(labels=cols_to_drop)

    for col in num_cols + cat_cols:
        if col in data.index:
            features.append(data[col])
        else:
            features.append(np.nan) 
    
    return np.array(features, dtype=float)

def concatenate_embeddings (head_emb, femur_emb, abdomen_emb, tabular_features): 

    multimodal_vector = np.concatenate ([head_emb, abdomen_emb, femur_emb, tabular_features])

    return multimodal_vector




df = pd.read_csv("../datasets/prospective-dataset/all_prospective_data.csv")

input_data = df.iloc[[10]] 

root_dir = "../datasets/prospective-dataset"

head_img_path = input_data[image_cols[1]].values[0]
femur_img_path = input_data[image_cols[2]].values[0]
abdomen_img_path = input_data[image_cols[0]].values[0]

head_img, head_valid = load_image (head_img_path, root_dir)
head_img=head_img.to(DEVICE)
femur_img, femur_valid = load_image(femur_img_path, root_dir)
femur_img=femur_img.to(DEVICE)
abdomen_img, abdomen_valid = load_image(abdomen_img_path, root_dir)
abdomen_img=abdomen_img.to(DEVICE)

pre_trained_model = load_multimodal_model (multimodal_model_path, num_cols, cat_cols)
tabular_features = get_tabular_features (input_data.iloc[0], num_cols, cat_cols)


head_emb = extract_image_embeddings (pre_trained_model,head_img, head_valid)
femur_emb = extract_image_embeddings (pre_trained_model,femur_img, femur_valid)
abdomen_emb = extract_image_embeddings (pre_trained_model,abdomen_img, abdomen_valid)

embedding_vector = concatenate_embeddings (head_emb, femur_emb, abdomen_emb, tabular_features)


