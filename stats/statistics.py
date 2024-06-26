import matplotlib.pyplot as plt
from collections import Counter


def pieChartClassificationPlot(sample, title):
    # ---------------------------------------------------
    # Create Pie Chart Plot for classification of samples
    # ---------------------------------------------------
    plt.pie(sample[1], labels=sample[0], autopct='%1.1f%%')
    plt.title(title)
    plt.show()

 
def barClassificationPlot(sample, title, x_label, y_label):
    # ---------------------------------------------
    # Create Bar Plot for classification of samples
    # ---------------------------------------------
    plt.bar(sample.keys(), sample.values())
    plt.title(title)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()


def ad_plot_bar(sample, title):
    # wrapper for barClassificationPlot
    class_counts = Counter(sample["y"])
    x_label = "CDR Rating"
    y_label = "Number of Images"

    barClassificationPlot(class_counts, title, x_label, y_label)



"""
class Statistics():
    def read_json(self, json_dir):
        json_data = {}

        with open(json_dir, 'r') as json_file:
            json_data = json.load(json_file)

        return json_data


    # explore OASIS-2 MRI's AD distribution using CDR
    def adClassificationDistribution(self, samples):
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
    

def main():
    # Run /preprocessing/nifti-json-selector.py first!
    # Make sure .json file is in /preprocessing folder!
    json_filename = f"sample_demographics.json" # example: "sample_demographics.json"

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.join(current_dir, "..")
    json_dir = f"{parent_dir}\\preprocessing\\{json_filename}"

    stats = Statistics()
    sample = stats.read_json(json_dir)
    sample = stats.adClassificationDistribution(sample)

    stats.pieChartClassificationPlot(sample)
"""