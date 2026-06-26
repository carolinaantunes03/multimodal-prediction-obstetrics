# Multimodal Machine Learning Models for Predicting Mode of Delivery After IOL

This project presents an evaluation of multimodal machine learning models for predicting the mode of delivery — vaginal delivery (VD) or cesarean section (CS) — following induction of labor (IOL). The models are trained and evaluated on two data modalities: tabular maternal-fetal clinical data and third-trimester ultrasound images from three anatomical views (abdomen, head, and femur). Multimodal architectures were trained using the retrospective dataset and evaluated on the prospective dataset. Multimodal model predictions were compared and combined with clinical practice.

## Repository Structure

- `*datasets*`: Datasets used in the experiments
- `*environments*`: Environment configuration files used for each model
- `*image-only*`: Image-only experiments
- `*tabular-only*`: Tabular-only experiments
- `*multimodal-architecture*`: Multimodal experiments
    - `*multimodal-architecture/models_prospective*`: Weights trained on retrospective data (for external validation)
    - `*multimodal-architecture/ensembles_experiment*`: Ensemble experiments combining model + clinical practice
- `*embeddings-optimisation` : Embedding optimisation experiments
- `*final_architecture*` : Code to run the final multimodal architecture
- `*results*` : Detailed results across all experimental stages
  
 Most directories contain their own *README.md* file with detailed explanations.


## Getting Started 

### Multimodal Experiments

The multimodal model uses an FT-Transformer as the tabular encoder and supports multiple image encoders. To run an experiment:

1. Clone the repository for your chosen image encoder (see table below)
2. Download the corresponding pretrained weights
3. Set up the associated conda environment
4. Update the required file paths in the config
5. Select the desired image encoder in `multimodal-architecture/models.py`


|Image Encoder  | Repository |Pre-Trained Weights  | Environment |
|--|--|--|--|
|**MedViT-V2**  | [GitHub](https://github.com/Omid-Nejati/MedViTV2.git) |[Download MedViT_base_Fetal.pth](https://drive.google.com/file/d/16bWPHWGQxvq_ynVYnRRfhANNMNlFx9O1/view) | [medvit2.yml](./environments/medvit2.yml) |
|**MedMamba**  | [GitHub](https://github.com/YubiaoYue/MedMamba.git) |[Download MedMamba.pth](https://zenodo.org/records/3904280) | [medmamba.yml](./environments/medmamba.yml) |
|**FetalCLIP**  | [GitHub](https://github.com/BioMedIA-MBZUAI/FetalCLIP.git)|[Download FetalCLIP_weights.pt](https://huggingface.co/numansaeed/fetalclip-model/blob/main/FetalCLIP_weights.pt)  | [fetalclip.yml](./environments/fetalclip.yml) |
|**MedSigLiP** | [HuggingFace](https://huggingface.co/google/medsiglip-448) *|- | [geral.yml](./environments/geral.yml) |
|**Swin Transformer** | - |- | [geral.yml](./environments/geral.yml). |


*MedSigLiP requires a Hugging Face account with write access. Generate a token [here](https://huggingface.co/settings/tokens). 

After that, execute the following code inside `*multimodal-architecture*`: 
```
conda env create -f <environment>.yml
conda activate <environmment>
```
For Internal Validation: 

```
python train_validate_multi_seeds_retrospective.py
```

For External Validation:
```
python external_validation.py
```
For Ensemble Experiments:
```
python ensemble_exteriment/ensembles_PC_multimodal.py
```


## Embedding Optimisation Experiments

After identifying the best-performing multimodal model, you can extract embeddings in multiple configurations running:

```
python multimodal_architecture/extract_embeddings/extract_embeddings.py
```

Alternatively, use the pretrained model weights from `*multimodal-architecture/models/prospective/*`.

### Steps:

1. Copy the resulting embeddings dataset to `*embeddings-optimisation/data/datasets/*`
2. Configure the EDCA algorithm parameters in `*embeddings-optimisation/benchmarks/test_config.json*`
3. Run the optimisation with 
```
docker compose -f docker-compose.benchmark.yml -p edca-benchmark up
```
4. Analyse results using the scripts in `*embeddings-optimisation/analysis/*`

