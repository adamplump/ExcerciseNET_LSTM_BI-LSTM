import os
import csv
import numpy as np
from collections import defaultdict

def sorting_folders(current_dir):
    folder = [f for f in os.listdir(current_dir)]
    folder_sorted = sorted(folder, key=lambda x: int(x[1:]))
    return folder_sorted


def load_labels(filepath):
    excercises_dict = defaultdict(list)
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            excercises_dict[line[3]].append(line[0:3])
        excercises_dict = dict(excercises_dict)
    return excercises_dict


def load_pose3d(file_path_3D) -> np.ndarray:
    try:
        mod = np.load(file_path_3D)
        mod = mod.astype(np.float32)
    except FileNotFoundError as e:
        print('{}. Returning None'.format(e))
        raise
    return mod


def sign_labels_to_workout(folder_list, current_dir):
    workouts = defaultdict(dict)
    for f in folder_list:
        labels = load_labels(os.path.join(current_dir, f, f + '_labels.csv'))
        workouts[f] = labels
        
    workouts = dict(workouts)
    return workouts

def sign_label_to_workout(workout, current_dir):
    label = load_labels(os.path.join(current_dir, workout, workout + '_labels.csv'))
    return label


def create_fp_pose3D(current_dir, workout):
    create_path = os.path.join(current_dir, str(workout), str(workout)+'_pose_3d.npy')
    return create_path