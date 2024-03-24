# Early, Accuracy, and Efficient Detection of Alzheimer's Disease through Artificial Intelligence Project
An AI Project for AI4ALL

## Requirements
- Python 3.10.5 or newer
- PyTorch
- scikit-image
- scikit-learn
- imbalanced-learn
- albumentations
- opencv2
- Pandas
- Numpy
- PIL (Pillow)
- Scipy
- NiBabel
- matplotlib
- openpyxl
- json

## Installation
Steps to running the model:

1. [Download](https://github.com/Verbosi7y/ai-alzheimer-detection/archive/refs/heads/main.zip) or run `git clone https://github.com/Verbosi7y/ai-alzheimer-detection.git`
2. If anything is modified or missing in `...\alzheimer-project-ai4all\assets\Kaggle\alzheimer_mri_preprocessed_dataset`, download the [Kaggle Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset).
   2a. If it is missing/modified, import the data from Kaggle into `...\assets\Kaggle\alzheimer_mri_preprocessed_dataset`.
   2b. Make sure the folders inside are labeled as `Non_Demented`, `Very_Mild_Demented`, `Mild_Demented`, `Moderate_Demented`.
5. Create folder called `resampled` in the root folder.

If any of these components or files are missing, the [repository](https://github.com/Verbosi7y/ai-alzheimer-detection) should contain the files necessary.

If the repository is private, please contact me through [GitHub](https://github.com/Verbosi7y).


## Usage
1. Run `01_preprocessing.ipynb` in `notebooks`
2. Follow the steps listed in the Notebook. Don't forget to enter the `parent_path`. I'm not exactly sure if it will run normally in MacOS and specifically, Apple Silicon.
3. (❌) Run '02_alzheimers-detection.ipynb`. We will be training the model in this notebook.
```

### License
Apache License 2.0

### README.md
Last Updated: March 24th, 2024
