'''
    Dataset.py -- Custom dataset file to load, save, and display images.
    Authors: Darwin Xue
'''
import numpy as np
import skimage
from PIL import Image

import os


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
        class_dir = fr"{parent_dir}\{class_name}"

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    for i, (image, label) in enumerate(zip(X, y)):
        img_class_path = fr"{parent_dir}\{classes[label]}\{i}.png"
        save_png_image(image, img_class_path)


def save_npz(X, y, npz_path):
    np.savez_compressed(npz_path, images=X, labels=y)


def save_images_npz(X, y, classes, parent_dir, npz_filename):
    npz_path = fr"{parent_dir}\{npz_filename}"
    save_images(X, y, classes, parent_dir)
    save_npz(X, y, npz_path)


def showImage(images, idx):
      skimage.io.imshow(images[idx])


def printShapes(images, labels):
    print("Images: ", images.shape)
    print("Labels: ", labels.shape)