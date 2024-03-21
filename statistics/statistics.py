import os
import json
import matplotlib.pyplot as plt

def read_json(json_dir):
    json_data = {}

    with open(json_dir, 'r') as json_file:
        json_data = json.load(json_file)

    return json_data


# explore MRI's AD distribution using CDR
def adClassificationDistribution(samples):
    ''' CDR:
            Healthy: 0.0
            Very Mild: 0.5
            Mild: 1
            Moderate: 2
            Severe: 3
    '''
    # Note: OASIS-2 does not contain any MRI imaging of Severe brains
    labels = ["Healthy", "Very Mild", "Mild", "Moderate"]
    sizes = [0, 0, 0, 0]

    for subject_id in samples:
        for mri_session in samples[subject_id]:
            cdr = samples[subject_id][mri_session]["CDR"]

            if cdr == 0:
                sizes[0] += 1

            if cdr == 0.5:
                sizes[1] += 1

            if cdr == 1.0:
                sizes[2] += 1

            if cdr == 2.0:
                sizes[3] += 1

    return (labels, sizes)


# CDR Distribution before data augmentation
def pieChartClassificationPlot(sample):
    classification = adClassificationDistribution(sample)

    plt.pie(classification[1], labels=classification[0], autopct='%1.1f%%')
    plt.title("AD Classification Distribution before Augmentation")
    plt.show()

# Run /preprocessing/nifti-json-selector.py first!
# Make sure .json file is in /preprocessing folder!
json_filename = f"ENTER JSON FILENAME" # example: "sample_demographics.json"

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, "..")
json_dir = f"{parent_dir}\\preprocessing\\{json_filename}"

sample = read_json(json_dir)

pieChartClassificationPlot(sample)
