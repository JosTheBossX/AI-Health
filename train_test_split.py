import argparse
import os
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from helper_code import load_patient_data, find_patient_files, get_murmur, get_outcome, get_weight, compare_strings, get_sex, get_age, get_height, get_pregnancy_status

def get_metadata(data):

    # Extract the age group and replace with the (approximate) number of months
    # for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, "Neonate"):
        age = 0.5
    elif compare_strings(age_group, "Infant"):
        age = 6
    elif compare_strings(age_group, "Child"):
        age = 6 * 12
    elif compare_strings(age_group, "Adolescent"):
        age = 15 * 12
    elif compare_strings(age_group, "Young Adult"):
        age = 20 * 12
    else:
        age = float("nan")

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, "Female"):
        sex_features[0] = 1
    elif compare_strings(sex, "Male"):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant]))

    return np.asarray(features, dtype=np.float32)


def train_test_split_def(
    list_features: list,
    data_directory: str,
    out_directory: str,
    test_size: float,
    random_state: int,
):
    # Check if out_directory directory exists, otherwise create it.
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    else:
        shutil.rmtree(out_directory)
        os.makedirs(out_directory)
    # Get metadata
    patient_files = find_patient_files(data_directory)
    num_patient_files = len(patient_files)
    murmur_classes = ["Present", "Unknown", "Absent"]
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ["Abnormal", "Normal"]
    num_outcome_classes = len(outcome_classes)
    features = list()
    murmurs = list()
    outcomes = list()
    for i in tqdm(range(num_patient_files)):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        # Extract features.
        current_features = get_metadata(current_patient_data)
        current_features = np.insert(
            current_features, 0, current_patient_data.split(" ")[0]
        )
        current_features = np.insert(
            current_features, 1, current_patient_data.split(" ")[2][:-3]
        )
        features.append(current_features)
        # Extract labels and use one-hot encoding.
        # Murmur
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)
        # Outcome
        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)
    features = np.vstack(features)
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)

    # Combine dataframes
    features_pd = pd.DataFrame(
        features,
        columns=[
            "id",
            "hz",
            "age",
            "female",
            "male",
            "height",
            "weight",
            "is_pregnant",
        ],
    )
    murmurs_pd = pd.DataFrame(murmurs, columns=murmur_classes)
    outcomes_pd = pd.DataFrame(outcomes, columns=outcome_classes)
    complete_pd = pd.concat([features_pd, murmurs_pd, outcomes_pd], axis=1)
    complete_pd["id"] = complete_pd["id"].astype(int).astype(str)
    # Split data
    complete_pd["stratify_column"] = (
        complete_pd[list_features].astype(str).agg("-".join, axis=1)
    )
    complete_pd_train, complete_pd_test = train_test_split(
        complete_pd,
        test_size=test_size,
        random_state=random_state,
        stratify=complete_pd["stratify_column"],
    )

    # Save the files.
    os.makedirs(os.path.join(out_directory, "train_data"))
    os.makedirs(os.path.join(out_directory, "test_data"))
    for f in complete_pd_train["id"]:
        copy_files(
            data_directory,
            f,
            os.path.join(out_directory, "train_data/"),
        )
    for f in complete_pd_test["id"]:
        copy_files(
            data_directory,
            f,
            os.path.join(out_directory, "test_data/"),
        )

def copy_files(data_directory: str, ident: str, out_directory: str) -> None:
    # Get the list of files in the data folder.
    files = os.listdir(data_directory)
    # Copy all files in data_directory that start with f to out_directory
    for f in files:
        if f.startswith(ident):
            _ = shutil.copy(os.path.join(data_directory, f), out_directory)


if __name__ == "__main__":

    list_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    # Create the test split.
    train_test_split_def(list_features, sys.argv[1], sys.argv[2],float(sys.argv[3]), int(sys.argv[4]))
    # data_directory , output_directory, test_size (0.2), random_state
    #python train_test_split.py training_data out_data 0.2 42
