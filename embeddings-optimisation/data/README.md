# Description

The datasets referenced in this repository correspond to embeddings extracted using our best-performing multimodal model and are not publicly available. To run the experiments provided in this repository, users must create their own dataset following a structure similar to the one described below.

## Directory Structure

Inside datasets/MedViT2-nopt/all-data/, create the following directories:

```
datasets/
│── MedViT2-nopt/
│ |── all-data/
│ │ |── prospective/
│ │ |── retrospective/

```

Each directory should contain .csv files corresponding to one of the following data configurations:

|Configuration  | Image Data | Tabular Data | Dimension |
|--|--| -- | --|
| i) | Averaged Embedding | Tabular Embedding | 960 |
| ii)| Averaged Embedding | Projected Tabular Embedding | 1536 |
| iii) | Averaged Embedding | Raw Tabular Features | 860 |
| iv) | 3 Individual Plane Embeddings | Raw Tabular Features | 2396 |


All .csv files must contain the following columns:

- Processo – unique patient/sample identifier
- Class – target label

## Feature Naming Conventions

To ensure compatibility with the codebase, feature columns must follow the naming conventions below:

|Feature Type  | Column Prefix | 
|--|--|
|Averaged Image Embeedings  | `img_emb` | 
|Tabular Embeedings  | `tab_emb` | 
|Abdomen Plane Embeedings  | `image_abdomen_emb` | 
|Femur Plane Embeedings  | `image_femur_emb`| 
|Head Plane Embeedings  | `image_head_emb`| 

Raw tabular features should retain their original feature names.

### Notes

- The dimensionality of each configuration must match the values specified in the table above.
- The repository assumes that embedding features are stored in separate columns using the prefixes described above.
- Any custom dataset should follow the same directory structure and column naming conventions to ensure compatibility with the provided scripts.