#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import MultiLayerPerceptron as mlp
from scipy.signal import spectrogram
from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    murmurs = list()
    outcomes = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract features.
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)

    features = np.vstack(features)
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Define parameters for random forest classifier.
    # n_estimators   = 123  # Number of trees in the forest.
    # max_leaf_nodes = 45   # Maximum number of leaf nodes in each tree.
    # random_state   = 6789 # Random state; set for reproducibility.

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    # features scaling, but using min-max scaling
    print(features)
    features_mean = np.mean(features, axis=0)
    print(features_mean)
    features_max_min = np.max(features, axis=0) - np.min(features, axis=0)
    print(features_max_min)
    features = (features - features_mean) / features_max_min
    print(features)
    print('/n/n')
    print(features.shape)
    murmur_classifier = mlp.Mlp([64],
                            features.shape[1], murmurs.shape[1],
                            mlp.relu, mlp.d_relu, mlp.softmax,
                            verbose=True).fit(features.T, murmurs.T, epochs=6000)

    outcome_classifier = mlp.Mlp([64],#20
                            features.shape[1], outcomes.shape[1],
                            mlp.relu, mlp.d_relu, mlp.softmax,
                            verbose=True).fit(features.T, outcomes.T, epochs=6000)
    
    # Save mean and max-min difference in the model (this will be used for pre-processing in the testing part.)
    murmur_classifier.data_store['mean'] = features_mean
    murmur_classifier.data_store['max-min'] = features_max_min

    outcome_classifier.data_store['mean'] = features_mean
    outcome_classifier.data_store['max-min'] = features_max_min

    # Save the model.
    save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    imputer = model['imputer']
    murmur_classes = model['murmur_classes']
    murmur_classifier = model['murmur_classifier']
    outcome_classes = model['outcome_classes']
    outcome_classifier = model['outcome_classifier']
    # current_features = get_features(current_patient_data, current_recordings)
  
    # Load features.
    features = get_features(data, recordings)
    # features.append(current_features)
    # Impute missing data.
    features = features.reshape(1, -1)
    features = imputer.transform(features)
    # features = np.vstack(features)
    # Retreive features scaling parameteres from the model (which was saved before)
    features_mean = murmur_classifier.data_store['mean']
    features_max_min = murmur_classifier.data_store['max-min']
    # Do features scaling
    features = (features - features_mean) / features_max_min

    # Get classifier probabilities.
    murmur_probabilities = murmur_classifier.predict_proba(features.T)
    murmur_probabilities = np.array(murmur_probabilities)
    outcome_probabilities = outcome_classifier.predict_proba(features.T)

    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier):
    d = {'imputer': imputer, 'murmur_classes': murmur_classes, 'murmur_classifier': murmur_classifier, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_classifier}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()
    num_spectrogram_features = 3
    spectrogram_features = np.zeros(num_spectrogram_features * num_recording_locations, dtype=float)
    for i, loc in enumerate(recording_locations):
        for j, rec_loc in enumerate(locations):
            if compare_strings(loc, rec_loc) and np.size(recordings[j]) > 0:
                rec = recordings[j]
                f, t, Sxx = spectrogram(rec, fs=4000, nperseg=256, noverlap=128)
                mean_Sxx = np.mean(Sxx)
                var_Sxx = np.var(Sxx)
                skew_Sxx = sp.stats.skew(Sxx.flatten())
                spectrogram_features[i * num_spectrogram_features:(i + 1) * num_spectrogram_features] = [mean_Sxx, var_Sxx, skew_Sxx]
    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features,spectrogram_features))

    return np.asarray(features, dtype=np.float32)
