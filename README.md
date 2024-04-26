# Early, Accuracy, and Efficient Detection of Alzheimer's Disease through Artificial Intelligence
An AI Project for AI4ALL

[![Click here to run on Google Colaboratory](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Verbosi7y/ai-alzheimer-detection/blob/main/notebooks/Alzheimer%20Detection%20Collab.ipynb)

## Requirements
- Python 3.10.5 or newer
- PyTorch
- Torchvision
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

*if you have an Nvidia and supports CUDA 12.1, run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` otherwise, run `pip3 install torchvision`


## Installation
Steps to running the model:

1. [Download](https://github.com/Verbosi7y/ai-alzheimer-detection/archive/refs/heads/main.zip) or run `git clone https://github.com/Verbosi7y/ai-alzheimer-detection.git`
2. If anything is modified or missing in `...\alzheimer-project-ai4all\assets\Kaggle\alzheimer_mri_preprocessed_dataset\raw`
   - Download the [Kaggle Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset).
   - If it is missing/modified, import the data from Kaggle into `...\assets\Kaggle\alzheimer_mri_preprocessed_dataset\raw`.
   - Make sure the folders inside are labeled as `Non_Demented`, `Very_Mild_Demented`, `Mild_Demented`, `Moderate_Demented`.

If any of these components or files are missing, the [repository](https://github.com/Verbosi7y/ai-alzheimer-detection) should contain the files necessary.

If the repository is private, please contact me through [GitHub](https://github.com/Verbosi7y).


## Usage
RUNNING LOCALLY:
1. Run `01_preprocessing.ipynb` in `notebooks`
2. Follow the steps listed in the Notebook. Don't forget to enter the `parent_path`. I'm not exactly sure if it will run normally in MacOS and specifically, Apple Silicon.
3. Run `02_alzheimers-detection.ipynb`. We will be training the model in this notebook.


## Credits
[Alzheimer MRI Preprocessed Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) by Sachin Kumar, Dr. Sourabh Shastri

[ADNI](https://adni.loni.usc.edu/) - Alzheimer's Disease Neuroimaging Initiative

[A novel CNN architecture for accurate early detection and classification of Alzheimer’s disease using MRI data](https://www.nature.com/articles/s41598-024-53733-6) by A.M. El-Assy, Hanan M. Amer, H. M. Ibrahim, M. A. Mohamed


Contributors
------------
- Developer: Verbosi7y


### License
Apache License 2.0

### README.md
Last Updated: April 26th, 2024
