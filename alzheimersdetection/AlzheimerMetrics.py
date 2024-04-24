'''
    AlzheimerMetric.py -- Determining accuracy of Alzheimer CNN model.
    Authors: Darwin Xue
'''
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

labels = ["Non_Demented",
          "Very_Mild_Demented",
          "Mild_Demented",
          "Moderate_Demented"]

def run_metrics(model, test_loader, device, classes=None):
    global labels
    
    all_preds = []
    all_labels = []

    if classes is None:
        classes = labels

    # Evaluation loop
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics for each class
    for class_index in range(4):  # Replace num_classes with the actual number of classes
        class_preds = (all_preds == class_index)
        class_labels = (all_labels == class_index)
        accuracy = accuracy_score(class_labels, class_preds)
        precision = precision_score(class_labels, class_preds, zero_division=0)
        recall = recall_score(class_labels, class_preds, zero_division=0)

        class_name = classes[class_index]

        print(f"{class_name}({class_index}) - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")