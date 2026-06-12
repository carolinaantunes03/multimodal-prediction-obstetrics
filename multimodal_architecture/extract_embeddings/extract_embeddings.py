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

import data
import models

experiment = "MedViT2_nopt"

dataset_retrospetivo = "../datasets/retrospective-dataset/all_retrospective_data.csv"
dataset_prospetivo = "../datasets/prospective-dataset/all_prospective_data.csv"


output_dir = Path("../datasets/embedding") / experiment / "all_data"
output_dir.mkdir(parents=True, exist_ok=True)


base_dir = Path(__file__).parent.resolve()
model_save_dir = base_dir / "models_prospective"
model_new_dir = model_save_dir / experiment
trained_model = model_new_dir / "final_model.pth"


with open("columns_config.json", "r") as f:
    col_config = json.load(f)

num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
images = ["abdomen_image", "head_image", "femur_image"]

processo = "Processo"
label = "Class"

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def to_2d(x):
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


def make_emb_cols(prefix, dim):
    # prefix already includes underscore if you want (e.g., "img_emb")
    return [f"{prefix}{i}" for i in range(1, dim + 1)]


def find_image_root(dataset_kind: str):
    """
    Attempts to find the correct image root folder.
    Adjust 'possible_roots' for your machines.
    """
    if dataset_kind == "retrospective":
        possible_roots = [
            "/home/beatrix/Documents/Carolina_A/tese-carolina/datasets/retrospective-dataset",
            "/home/carolantunes/datasets/retrospective-dataset",
            "../datasets/retrospective-dataset",
        ]
    else:
        possible_roots = [
            "/home/beatrix/Documents/Carolina_A/tese-carolina/datasets/prospective-dataset",
            "/home/carolantunes/datasets/prospective-dataset",
            "../datasets/prospective-dataset",
        ]

    # Simple existence check; adapt if your structure differs
    for root in possible_roots:
        if os.path.exists(root):
            return root

    raise FileNotFoundError(f"Could not find image root for {dataset_kind}. Update possible_roots.")


def build_model(train_like_df_for_categories: pd.DataFrame):
    
    num_numerical = len(num_cols)
    num_categories = [train_like_df_for_categories[c].nunique() + 1 for c in cat_cols]
    tabular_token_dim = 192
    tabular_hidden_dim = 192

    model = models.build_multimodal_model(
        num_numerical, num_categories, tabular_token_dim, tabular_hidden_dim
    ).to(DEVICE)

    print(f"Loading weights: {trained_model}")
    model.load_state_dict(torch.load(trained_model, map_location=DEVICE))
    model.eval()
    return model


def build_preprocessor(fit_df: pd.DataFrame):
    tab_preproc = data.TabularPreprocessor(num_cols, cat_cols)
    tab_preproc.fit(fit_df)
    return tab_preproc


def build_loader(df: pd.DataFrame, tab_preproc, image_cols, image_root, batch_size=16):
    ds = data.MultimodalDataset(
        df,
        tab_preproc,
        image_cols=image_cols,
        y_col=label,
        train=False,
        root_dir=image_root,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def extract_raw_and_projected_embeddings(loader, model, device, print_shapes=False):
    """
    Extract:
      - raw_tab embeddings
      - img embeddings
      - projected tab embeddings
    """
    model.eval()

    y_all = []
    raw_tab_all = []
    raw_img_all = []
    proj_tab_all = []


    printed = False

    with torch.no_grad():
        for b, batch in enumerate(loader):
            print(f"  ... batch {b+1}/{len(loader)}", end="\r")

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            labels = batch["label"]

            raw_tab, _ = model.fttransformer(batch, return_feature=True)
            raw_img, _ = model.image_encoder(
                batch["images"],
                image_valid_num=batch.get("image_valid_num"),
                return_feature=True
            )

            raw_tab = F.layer_norm(raw_tab, raw_tab.shape[1:])
            raw_img = F.layer_norm(raw_img, raw_img.shape[1:])

            proj_tab = model.fusion_mlp.tabular_adapter(raw_tab)
           

            if print_shapes and not printed:
                print("\nEmbedding vector sizes (per sample):")
                print("  raw tabular:       ", raw_tab.shape[1:])
                print("  raw image:         ", raw_img.shape[1:])
                print("  projected tabular: ", proj_tab.shape[1:])
               
                printed = True

            y_all.append(labels.detach().cpu().numpy())
            raw_tab_all.append(raw_tab.detach().cpu().numpy())
            raw_img_all.append(raw_img.detach().cpu().numpy())
            proj_tab_all.append(proj_tab.detach().cpu().numpy())
            

 

    y_all = np.concatenate(y_all, axis=0)
    raw_tab_all = np.concatenate(raw_tab_all, axis=0)
    raw_img_all = np.concatenate(raw_img_all, axis=0)
    proj_tab_all = np.concatenate(proj_tab_all, axis=0)
    

    return y_all, raw_tab_all, raw_img_all, proj_tab_all


def extract_image_embeddings_only(loader, model, device):
    """
    Extract only raw image embeddings.
    """
    model.eval()
    img_all = []
    y_all = []

    with torch.no_grad():
        for b, batch in enumerate(loader):
            print(f"  ... batch {b+1}/{len(loader)}", end="\r")

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            y_all.append(batch["label"].detach().cpu().numpy())

            raw_img, _ = model.image_encoder(
                batch["images"],
                image_valid_num=batch.get("image_valid_num"),
                return_feature=True
            )
            raw_img = F.layer_norm(raw_img, raw_img.shape[1:])
            img_all.append(raw_img.detach().cpu().numpy())

    print("\n  ... done.")
    y_all = np.concatenate(y_all, axis=0)
    img_all = np.concatenate(img_all, axis=0)
    return y_all, img_all


# Save fusion of embeddings 

def save_tab_emb_only(df_meta, tab_emb, out_path, prefix="tab_emb"):
    tab_emb = to_2d(tab_emb)
    tab_cols = make_emb_cols(prefix, tab_emb.shape[1])

    out = pd.DataFrame(tab_emb, columns=tab_cols)
    out.insert(0, processo, df_meta[processo].values)
    out[label] = df_meta[label].values

    out.to_csv(out_path, index=False)


def save_img_emb_only(df_meta, img_emb, out_path, prefix="img_emb"):
    img_emb = to_2d(img_emb)
    img_cols = make_emb_cols(prefix, img_emb.shape[1])

    out = pd.DataFrame(img_emb, columns=img_cols)
    out.insert(0, processo, df_meta[processo].values)
    out[label] = df_meta[label].values

    out.to_csv(out_path, index=False)


def save_fusion_emb(df_meta, tab_emb, img_emb, out_path, tab_prefix="tab_emb", img_prefix="img_emb"):
    tab_emb = to_2d(tab_emb)
    img_emb = to_2d(img_emb)

    tab_cols = make_emb_cols(tab_prefix, tab_emb.shape[1])
    img_cols = make_emb_cols(img_prefix, img_emb.shape[1])

    fused = np.concatenate([img_emb, tab_emb], axis=1)  
    out = pd.DataFrame(fused, columns=img_cols + tab_cols)

    out.insert(0, processo, df_meta[processo].values)
    out[label] = df_meta[label].values
    out.to_csv(out_path, index=False)


def save_image_plus_raw_features(df_meta, img_emb, out_path, img_prefix="img_emb"):
    """
    [img_emb1..N, <raw tabular features with their actual names>]
   .
    """
    img_emb = to_2d(img_emb)
    img_cols = make_emb_cols(img_prefix, img_emb.shape[1])

    # raw tabular "real" features (no embeddings)
    feature_cols = [c for c in (num_cols + cat_cols) if c in df_meta.columns]

    out = pd.DataFrame(img_emb, columns=img_cols)
    for c in feature_cols:
        out[c] = df_meta[c].values

    out.insert(0, processo, df_meta[processo].values)
    out[label] = df_meta[label].values
    out.to_csv(out_path, index=False)

def save_femur_head_abdomen_plus_raw_features(
    df_meta, femur_emb, head_emb, abdomen_emb, out_path,
    femur_prefix="image_femur_emb",
    head_prefix="image_head_emb",
    abdomen_prefix="image_abdomen_emb",
):
    femur_emb = to_2d(femur_emb)
    head_emb = to_2d(head_emb)
    abdomen_emb = to_2d(abdomen_emb)

    femur_cols = make_emb_cols(femur_prefix, femur_emb.shape[1])
    head_cols = make_emb_cols(head_prefix, head_emb.shape[1])
    abdomen_cols = make_emb_cols(abdomen_prefix, abdomen_emb.shape[1])

    # raw tabular "real" features (no embeddings)
    feature_cols = [c for c in (num_cols + cat_cols) if c in df_meta.columns]

    fused = np.concatenate([femur_emb, head_emb, abdomen_emb], axis=1)
    out = pd.DataFrame(fused, columns=femur_cols + head_cols + abdomen_cols)

    # append raw tabular features
    for c in feature_cols:
        out[c] = df_meta[c].values

    out.insert(0, processo, df_meta[processo].values)
    out[label] = df_meta[label].values
    out.to_csv(out_path, index=False)




def run_for_dataset(dataset_kind: str, csv_path: str, tab_preproc, model, out_dir: Path, print_shapes=False):
    """
    Creates ALL requested datasets for retrospective OR prospective.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Loading {dataset_kind} data: {csv_path} ===")
    full_df = pd.read_csv(csv_path)

    # Keep Processo + Class for metadata, but exclude Processo from model input
    if processo not in full_df.columns:
        raise ValueError(f"{processo} column not found in {csv_path}")
    if processo not in full_df.columns:
        raise ValueError(f"{label} column not found in {csv_path}")

    df_meta = full_df[[processo, label] + [c for c in (num_cols + cat_cols) if c in full_df.columns]].copy()

    # Model input dataframe: drop Processo, keep Class (dataset uses y_col=Class)
    model_df = full_df.drop(processo, axis=1, errors="ignore").reset_index(drop=True)
    df_meta = df_meta.reset_index(drop=True)

    image_root = find_image_root(dataset_kind)
    print(f"Using image root: {image_root}")

    # ----- A) embeddings with ALL images together (MultimodalDataset will handle list of cols)
    print(f"\n--- Extracting joint-image embeddings ({dataset_kind}) ---")
    loader_all_imgs = build_loader(model_df, tab_preproc, images, image_root, batch_size=16)

    y, raw_tab, raw_img, proj_tab, proj_img = extract_raw_and_projected_embeddings(
        loader_all_imgs, model, DEVICE, print_shapes=print_shapes
    )


    # 1) raw image + raw tab embeddings fusion 
    save_fusion_emb(
        df_meta, raw_tab, raw_img,
        out_dir / f"{dataset_kind}_fusion_raw_img_raw_tab.csv",
        tab_prefix="tab_emb", img_prefix="img_emb"
    )

    # 2) raw image + projected tab embeddings
    save_fusion_emb(
        df_meta, proj_tab, raw_img,
        out_dir / f"{dataset_kind}_fusion_raw_img_projected_tab.csv",
        tab_prefix="tab_emb", img_prefix="img_emb"
    )

    # 3) fusion of image embeddings with REAL tabular features
    save_image_plus_raw_features(
        df_meta, raw_img,
        out_dir / f"{dataset_kind}_fusion_img_emb_plus_raw_features.csv",
        img_prefix="img_emb"
    )

    # 4) raw tabular embeddings alone
    save_tab_emb_only(
        df_meta, raw_tab,
        out_dir / f"{dataset_kind}_raw_tab_emb.csv",
        prefix="tab_emb"
    )

    # 5) projected tabular embeddings alone
    save_tab_emb_only(
        df_meta, proj_tab,
        out_dir / f"{dataset_kind}_projected_tab_emb.csv",
        prefix="tab_emb"
    )

    # 6) image embeddings alone (joint / as used in your model)
    save_img_emb_only(
        df_meta, raw_img,
        out_dir / f"{dataset_kind}_raw_img_emb.csv",
        prefix="img_emb"
    )

    # ----- B) per-plane image embeddings alone
    print(f"\n--- Extracting per-plane image embeddings ({dataset_kind}) ---")
    plane_map = {
        "abdomen": ["abdomen_image"],
        "head": ["head_image"],
        "femur": ["femur_image"],
    }
    abdomen_emb = None
    head_emb = None
    femur_emb = None
    
    for plane_name, plane_cols in plane_map.items():
        

        # only run if those columns exist
        missing = [c for c in plane_cols if c not in model_df.columns]
        if missing:
            print(f"Skipping plane '{plane_name}' (missing columns: {missing})")
            continue

        print(f"  Plane: {plane_name}")
        loader_plane = build_loader(model_df, tab_preproc, plane_cols, image_root, batch_size=16)
        y_plane, img_plane = extract_image_embeddings_only(loader_plane, model, DEVICE)

        if len(y_plane) != len(df_meta):
            raise RuntimeError(
                f"Per-plane label length ({len(y_plane)}) != df length ({len(df_meta)}). Dataset may be filtering rows."
            )
        
        if plane_name == "abdomen":
            abdomen_emb = img_plane
        elif plane_name == "head":
            head_emb = img_plane
        elif plane_name == "femur":
            femur_emb = img_plane


        save_img_emb_only(
            df_meta, img_plane,
            out_dir / f"{dataset_kind}_raw_{plane_name}_img_emb.csv",
            prefix=f"image_{plane_name}_emb"
        )

    if femur_emb is not None and head_emb is not None and abdomen_emb is not None:
        save_femur_head_abdomen_plus_raw_features(
        df_meta=df_meta,
        femur_emb=femur_emb,
        head_emb=head_emb,
        abdomen_emb=abdomen_emb,
        out_path=out_dir / f"{dataset_kind}_femur_head_abdomen_emb_plus_raw_features.csv",
        femur_prefix="image_femur_emb",
        head_prefix="image_head_emb",
        abdomen_prefix="image_abdomen_emb",
        )
        
        

    print(f"\nSaved outputs to: {out_dir}")


def main():
    # Load both CSVs
    retro_df = pd.read_csv(dataset_retrospetivo)
    prosp_df = pd.read_csv(dataset_prospetivo)

    # Fit preprocessor
    # Recommended: fit on retrospective (often your "training domain"), but you can switch to concat if you prefer:
    # fit_df = pd.concat([retro_df, prosp_df], ignore_index=True)
    fit_df = retro_df.drop(processo, axis=1, errors="ignore")
    tab_preproc = build_preprocessor(fit_df)

    # Build model (category sizes based on fit_df)
    model = build_model(fit_df)

    # Run for both datasets
    run_for_dataset(
        dataset_kind="retrospective",
        csv_path=dataset_retrospetivo,
        tab_preproc=tab_preproc,
        model=model,
        out_dir=output_dir / "retrospective",
        print_shapes=True
    )

    run_for_dataset(
        dataset_kind="prospective",
        csv_path=dataset_prospetivo,
        tab_preproc=tab_preproc,
        model=model,
        out_dir=output_dir / "prospective",
        print_shapes=False
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
