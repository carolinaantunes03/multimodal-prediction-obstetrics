
# Dataset Requirements

The dataset used in this study is **not publicly available**. To run the code in this repository, you must provide your own dataset with a similar structure and characteristics.

---

## Data Modalities

Your dataset must include **two modalities**:

### 1. Tabular Clinical Data
- Contains **numerical and categorical variables**
- Feature names and expected structure are defined in: `columns_config.json`

### 2. Ultrasound Images 
- Third-trimester scans
- Three anatomical planes per patient:
    - Head
    - Abdomen
    - Femur
- Image format: .png

## Required Datasets

You must prepare **two separate datasets**:

### Retrospective Dataset (Model Development)

Used for training and validation.

- Must be pre-split into 3 cross-validation folds
- Each fold should include:
    - Training set
    - Validation set
    - Test set

### Prospective Dataset (Model Evaluation)

Used only for final testing.

- No splitting required
- Should remain completely independent from training data

## Folder Structure

Each dataset should be stored in a separate folder with the following structure:

prospective-dataset/
│── all_prospective_data.csv
│── PT_data_processed.csv      (Clinical Practice values)
│── prospective_images/
│ |── processo/
│ │ |── abdomen.png
│ │ |── head.png
│ │ └── femur.png
│ └── ...

retrospective-dataset/
│── all_retrospective_data.csv
│── images/
│ |── abdomen/
│ │  |── processo.png
│ │ └── ...
│ |── head/
│ │ |── processo.png
│ │ └── ...
│ |── femur/
│ │ |── processo.png
│ │ └── ...
│
│── cv1/
│ |── images_cv1/
│ |   |── all_cv1_train.csv
│ |   |── all_cv1_validation.csv
│ | └── all_cv1_test.csv
│
│── cv2/
│ └── ...
│
│── cv3/
│ └── ...

## CSV File Requirements

The data.csv file must include:

- All tabular clinical variables 
- A column with relative paths to the images
- Column names that exactly match those defined in `columns_config.json`
