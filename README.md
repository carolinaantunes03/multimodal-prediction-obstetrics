# Multimodal Machine Learning Models for Predicting Mode of Delivery After IOL

This project presents an evaluation of multimodal machine learning models for predicting the mode of delivery — vaginal delivery (VD) or cesarean section (CS) — following induction of labor (IOL). The models are trained and evaluated on two data modalities: tabular maternal-fetal clinical data and third-trimester ultrasound images from three anatomical views (abdomen, head, and femur). Multimodal architectures were trained using the retrospective dataset and evaluated on the prospective dataset. Multimodal model predictions were compared and combined with clinical practice.

## Repository Structure

- `*datasets*`: Datasets used in the experiments
- `*environments*`: Environment configuration files used for each model
- `*image-only*`: Code for image-only experiments
- `*multimodal-architecture*`: Code for multimodal experiments
    - `*multimodal-architecture/checkpoints_external_validation*`: Links to model weights trained on the retrospective dataset for external validation
    - `*multimodal-architecture/checkpoints_internal_validation*`: Links to best-run model weights from retrospective dataset training using 3-fold cross-validation 
    - `multimodal-architecture/ensembles_experiment*`: Code for ensembles experiments with clinical practice
- `*tabular-only*`: Code for tabular-only experiments
  
### Notes
1. Most directories contain their own *README.md* file with detailed explanations.


## Getting Started

This project uses state-of-the-art models as both image encoders and tabular encoders.

The multimodal model employs an FTTransformer as the tabular encoder and supports different choices of image encoders.

For each image encoder, you must clone the corresponding model repository and use its associated environment configuration file.

To run an experiment, you only need to:

- update the required file paths
- select the desired image encoder in multimodal-architecture/models.py.


### Image Encoders, Weights and Environments 
1. **MedViTV2**

Clone this github repository: https://github.com/Omid-Nejati/MedViTV2.git

The pretrained FetalCLIP model can be downloaded from the following link:
[Download MedViT_base_Fetal.pth](https://drive.google.com/file/d/16bWPHWGQxvq_ynVYnRRfhANNMNlFx9O1/view)

To run the model with MedViTV2 as the image encoder, use the environment file [medvit2.yml](./environments/medvit2.yml).

2. **MedMamba**

Clone this github repository: https://github.com/YubiaoYue/MedMamba.git

The pretrained FetalCLIP model can be downloaded from the following link:

[Download MedMamba.pth](https://zenodo.org/records/3904280)

To run the model with MedMamba as the image encoder, use the environment file [medmamba.yml](./environments/medmamba.yml).

3. **FetalCLIP**

Clone this github repository: https://github.com/BioMedIA-MBZUAI/FetalCLIP.git

The pretrained FetalCLIP model can be downloaded from the following link:

[Download FetalCLIP_weights.pt](https://huggingface.co/numansaeed/fetalclip-model/blob/main/FetalCLIP_weights.pt)

To run the model with FetalCLIP as the image encoder, use the environment file [fetalclip.yml](./environments/fetalclip.yml).


4. **MedSigLiP**

To run the model with MedSigLiP as the image encoder, use the environment file [geral.yml](./environments/geral.yml).
You also need access to MedSigLiP models on Hugging Face: https://huggingface.co/google/medsiglip-448
Generate a Hugging Face write access token by going to [settings](https://huggingface.co/settings/tokens). 

5. **Swin Transformer**

To run the model with Swin Transformer as the image encoder, use the environment file [geral.yml](./environments/geral.yml).

