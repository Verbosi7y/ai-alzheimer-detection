import json
import pandas as pd


def read_excel(excel_dir):
    excel = pd.read_excel(excel_dir)
    return excel


def write_json(samples, json_dir):
    json_data = json.dumps(samples, indent=4)

    with open(json_dir, 'w') as json_file:
        json_file.write(json_data)


def dataframe_to_dict(df, iter, allowConverted=False):
    if iter < 0:
        print("error: iter is not valid")
        return {}
    
    samples = {}

    for i, row in df.iterrows():
        if not iter: break
        if not allowConverted and row['Group'] == 'Converted': continue

        subject_id = row['Subject ID']
        mri_id = row["MRI ID"]
        
        # row = row.drop('Subject ID')
        # row = row.drop("MRI ID")
        row = row.to_dict()

        if subject_id not in samples:
            samples[subject_id] = {}

        samples[subject_id][mri_id] = row

        iter-=1

    if iter > 0:
        print("warning: iter is greater than available subjects")
    
    return samples

if __name__ == "__main__":
    longitudinal_filename = "oasis_longitudinal_demographics.xlsx"
    json_dir = "sample_demographics.json"
    sample_count = 5

    excel_df = read_excel(longitudinal_filename)
    samples = dataframe_to_dict(excel_df, sample_count)

    write_json(samples, json_dir)

