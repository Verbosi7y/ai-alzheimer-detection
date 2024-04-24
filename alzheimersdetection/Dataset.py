'''
    Dataset.py -- Custom dataset file to load, save, and display images.
    Authors: Darwin Xue
'''
import numpy as np

from PIL import Image
import cv2
import skimage
import albumentations as aug

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN

import os

labels = ["Non_Demented",
          "Very_Mild_Demented",
          "Mild_Demented",
          "Moderate_Demented"]

encoded_labels = {status: idx for idx, status in enumerate(labels)}


def step1_load_data(path):
    X, y = load_dataset(labels, path)
    printShapes(X, y)

    return X, y


def step2_split_data(X, y, test_size=0.20, validation_size=0.25):
    split_dataset = {
                        "train":        {
                                            "X": [],
                                            "y": []
                                        },
                        "test":         {
                                            "X": [],
                                            "y": []
                                        },
                        "validation":   {
                                            "X": [],
                                            "y": []
                                        }
                    }
    
    X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=test_size, stratify=y)
    
    print("Before Validation - Training Data Shape:", X_train.shape)
    print("Before Validation - Training Label Shape:", y_train.shape)

    X_train, X_val, y_train, y_val = train_test_split(X_train,  y_train, test_size=validation_size, stratify=y_train)

    # ----------------------------------------------------
    # Encoding Training, Testing, and Validation into Dict
    # ----------------------------------------------------
    split_dataset["train"]["X"] = X_train
    split_dataset["train"]["y"] = y_train

    split_dataset["test"]["X"] = X_test
    split_dataset["test"]["y"] = y_test

    split_dataset["validation"]["X"] = X_val
    split_dataset["validation"]["y"] = y_val

    display_split(split_dataset=split_dataset)

    return split_dataset


def step3a_augmentation(train, rates):
    X = train["X"]
    y = train["y"]

    X_aug, y_aug = [], []

    data_transforms = aug.Compose(
        [
            aug.Resize(height=128, width=128),
            aug.HorizontalFlip(p=0.5),
            aug.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            aug.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.4, p=1)
        ]
    )

    for i, (image, label) in enumerate(zip(X, y)):
        for j in range(rates[label]):
            augment_image = augment(image, data_transforms)
            X_aug.append(augment_image)
            y_aug.append(label)

    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)

    train["X"] = np.concatenate((X, X_aug), axis=0)
    train["y"] = np.concatenate((y, y_aug), axis=0)

    return train


def step3b_ADASYN(sample, k=5):
    X = sample["X"]
    y = sample["y"]

    # Preprocess data (normalize pixel values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X.reshape(-1, X.shape[1] * X.shape[2]))  # Reshape for normalization

    adasyn = ADASYN(n_neighbors=k)

    X, y = adasyn.fit_resample(X, y)

    # Reshape the 2d np array back to a 3d np array
    size = X.shape[0]
    X = X.reshape(size, 128, 128)

    # Invert normalization
    # Scale back to 0-255 and convert to uint8 for PyTorch
    X = (X * 255).astype(np.uint8)

    # Display a sampled MRI scan of the brain
    print(X.shape)
    showImage(X, len(X)-1)

    sample["X"] = X
    sample["y"] = y

    return sample


def step4_save_npz(sample, path):
    train_dir = os.path.join(path, "Train_Data")
    train_npz = "augmented_adasyn_train_data.npz"

    validation_dir = os.path.join(path, "Validation_Data")
    validation_npz = "val_data.npz"

    test_dir = os.path.join(path, "Test_Data")
    test_npz = "test_data.npz"

    X_train = sample["train"]["X"]
    y_train = sample["train"]["y"]

    X_test = sample["train"]["X"]
    y_test = sample["train"]["y"]

    X_val = sample["train"]["X"]
    y_val = sample["train"]["y"]

    print("[Training Dataset]: Saving...");
    save_images_npz(X_train, y_train, labels, train_dir, train_npz)
    print("[Training Dataset]: Done");

    print("[Testing Dataset]: Saving...");
    save_images_npz(X_test, y_test, labels, test_dir, test_npz)
    print("[Testing Dataset]: Done");

    print("[Validation Dataset]: Saving...");
    save_images_npz(X_val, y_val, labels, validation_dir, validation_npz)
    print("[Validation Dataset]: Done");



def augment(image, transform):
    augmented_image = transform(image=np.array(image))["image"]  # Extract augmented image from Albumentations output
    return augmented_image


def distribution(y):
    unique = np.unique(y, return_counts=True)
    size = unique[1].tolist()

    sample_dist = (labels, size)

    print(encoded_labels)
    print(unique)

    return sample_dist


def display_split(split_dataset):
    X_train = split_dataset["train"]["X"]
    y_train = split_dataset["train"]["y"]

    X_test = split_dataset["test"]["X"]
    y_test = split_dataset["test"]["y"]

    X_val = split_dataset["validation"]["X"]
    y_val = split_dataset["validation"]["y"]

    # See the overall sizes and verify the split occured correctly for each classes
    print("Training Size: ", X_train.shape)
    print("Test Size: ", X_test.shape)
    print("Validation Size: ", X_val.shape)

    print("\nClasses encoded for reference: ", encoded_labels, "\n")

    unique = np.unique(y_train, return_counts=True)
    print("Training Split: ", unique)

    unique = np.unique(y_val, return_counts=True)
    print("Validation Split: ", unique)

    unique = np.unique(y_test, return_counts=True)
    print("Testing Split: ", unique)


def load_dataset(classes, load_dir):
    X, y = [], []
    for class_name in classes:
        class_path = os.path.join(load_dir, class_name)

        for image_filename in os.listdir(class_path):
            img = skimage.io.imread(os.path.join(class_path, image_filename), as_gray=True)
            X.append(img)
            # 0 = Non Demented, 1 = Very Mild Demented, 2 = Mild Demented, 3 = Moderate Demented
            y.append(classes.index(class_name))
    
    return np.array(X), np.array(y)

    
def save_png_image(image, output_dir, format="PNG"):
    image = Image.fromarray(image)
    image.save(output_dir, format=format)


def save_images(X, y, classes, parent_dir):
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    for class_name in classes:
        class_dir = os.path.join(parent_dir, class_name)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    for i, (image, label) in enumerate(zip(X, y)):
        class_dir = os.path.join(parent_dir, classes[label])
        img_path = os.path.join(class_dir, f"{i}.png")
        save_png_image(image, img_path)


def save_npz(X, y, npz_path):
    np.savez_compressed(npz_path, images=X, labels=y)


def save_images_npz(X, y, classes, parent_dir, npz_filename):
    npz_path = os.path.join(parent_dir, npz_filename)
    save_images(X, y, classes, parent_dir)
    save_npz(X, y, npz_path)


def showImage(images, idx):
      skimage.io.imshow(images[idx])


def printShapes(images, labels):
    print("Images: ", images.shape)
    print("Labels: ", labels.shape)