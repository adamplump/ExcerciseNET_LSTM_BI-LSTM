import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict
import os
import csv
import utility
import plotting3D
from matplotlib import pyplot as plt
from features_extractor import FeatureExtractor
from Model.modelLSTM import RecurrentAutoencoder
from Model.modelCondLSTM import RecurrentAutoencoderCond
from Model.modelCondBiLSTM import RecurrentAutoencoderCondBi
from Model.modelCondBiLSTMdropout import RecurrentAutoencoderCondBiDropout

from Dataset.DatasetExc import ExcDataset
from Dataset.Transform import Transform
import yaml
import seaborn as sns
import pandas as pd
from scipy.ndimage import convolve1d
from sklearn.metrics import precision_score, recall_score, accuracy_score
import random

with open('configs/config_Adam.yaml', 'r') as file:
# with open('configs/config_Artur.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
    
nodes = { 'Hip': 0, 'Left Hip': 1 , 'Left Knee': 2, 'Left Foot':3, 'Right Hip': 4, 'Right Knee': 5 ,
        'Right Foot': 6, 'Spine': 7 , 'Thorax': 8, 'Neck/Nose': 9 , 'Head': 10 , 'Right Shoulder': 11, 
        'Right Elbow': 12, 'Right Wrist': 13 , 'Left Shoulder': 14, 'Left Elbow': 15, 'Left Wrist': 16}

limbs = [(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16), (0, 4), (4, 5), (5, 6), (8, 11), (11, 12),
             (12, 13), (8, 9), (9, 10), (0, 7), (7, 8)] # Kończyny stworzone z keypointów

limbs_angle = [(13, 12, 11), (16, 15, 14), (12, 11, 8), (15, 14, 8), (5, 4, 0), (2, 1, 0), (5, 0, 2)
               , (0, 7, 8), (7, 8, 9), [6, 5, 4], [3, 2, 1]] # Keypointy między, którymi obliczany jest kąt

sym_plane = [(0,7,9)] # Keypointy, które służą jako płaszczyzna symetrii

sym_limbs = [(0,6), (1,7), (2,8), (3,9), (4,10), (5,11)] # Odnosi się do limbs - czyli (0,6) oznacza kończyne: limbs[0]->(0,1) oraz limbs[1]->(0,4)

g_nodes = [2, 5, 7, 8, 10, 12, 13 , 15, 16] # Keypointy między, którymi obliczany jest kąt od center of mass do contact points

contact_points = [0, 1, 4] # Keypointy, na których opiera się ciężar ciała

gender = 'male'

excercises_class = {'squats': 0, 'pushups': 1, 'dumbbell_shoulder_press': 2, 'lunges': 3,
                  'dumbbell_rows': 4, 'situps': 5, 'tricep_extensions': 6, 'bicep_curls': 7,
                  'lateral_shoulder_raises': 8, 'jumping_jacks': 9, 'non_activity': 10} # Typy ćwiczeń wszystkie


# exercises_class_touse = {'squats': 0.011, 'dumbbell_shoulder_press': 0.018, 'lunges': 0.013,
#                   'dumbbell_rows': 0.0188, 'situps': 0.0225, 'tricep_extensions': 0.024, 
#                   'bicep_curls': 0.0188, 'lateral_shoulder_raises': 0.013, 'jumping_jacks': 0.012}

# #   BiLSTM 30 seq
# exercises_class_touse = {'squats': 0.019, 'dumbbell_shoulder_press': 0.026, 'lunges': 0.025,
#                   'dumbbell_rows': 0.028, 'situps': 0.03, 'tricep_extensions': 0.024, 
#                   'bicep_curls': 0.022, 'lateral_shoulder_raises': 0.023, 'jumping_jacks': 0.019}

#   BiLSTM 60 seq
exercises_class_touse = {'squats': 0.027, 'dumbbell_shoulder_press': 0.034, 'lunges': 0.032,
                  'dumbbell_rows': 0.041, 'situps': 0.049, 'tricep_extensions': 0.035, 
                  'bicep_curls': 0.042, 'lateral_shoulder_raises': 0.036, 'jumping_jacks': 0.028}

# #   BiLSTM 10 seq
# exercises_class_touse = {'squats': 0.06, 'dumbbell_shoulder_press': 0.06, 'lunges': 0.06,
#                   'dumbbell_rows': 0.06, 'situps': 0.06, 'tricep_extensions': 0.06, 
#                   'bicep_curls': 0.06, 'lateral_shoulder_raises': 0.06, 'jumping_jacks': 0.06}

# #   LSTM 10 seq
# exercises_class_touse = {'squats': 0.035, 'dumbbell_shoulder_press': 0.040, 'lunges': 0.037,
#                   'dumbbell_rows': 0.043, 'situps': 0.043, 'tricep_extensions': 0.038, 
#                   'bicep_curls': 0.043, 'lateral_shoulder_raises': 0.040, 'jumping_jacks': 0.039}


TRAIN_W_IDs = config['DATASET']['TRAIN_W_IDs']
VAL_W_IDs = config['DATASET']['VAL_W_IDs']
TEST_W_IDs = config['DATASET']['TEST_W_IDs']

batch_sizes = config['TRAIN']['BATCH_SIZE']
n_features = config['MODEL']['INPUT_DIM']
seq_len = config['DATASET']['SEQLEN']

embedding_dim = config['MODEL']['EMBEDDING_DIMS']
condition_dim = config['MODEL']['CONDITION_DIM']

current_dir = config['CURRENT_DIR']
models_dir = config['MODELS']
loss_plots_dir = config['LOSS_PLOTS']

plot_losses = True
bad_habits_probability = 1
workout_in_folder = utility.sorting_folders(current_dir=current_dir)
workout = 'w01'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RecurrentAutoencoderCondBiDropout(seq_len, n_features, embedding_dim, condition_dim, dropout_rate=0.0) 
# model = RecurrentAutoencoderCond(seq_len, n_features, embedding_dim, condition_dim,) 
model.to(device=device)
# checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0005seq30edim512ep100.pth"), 
# checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq10edim128eps304/bs64lr0.0001seq10edim128ep291drop0.3.pth"),     #   top z dropout v2
# checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq10edim128eps303/bs64lr0.0001seq10edim128ep203drop0.03.pth"),      #   najlepszy z dropoutem na cechach
# checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq10edim256eps301/bs64lr0.  0001seq10edim256ep300.pth"),    #   BiLSTM
# checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq10edim128eps300/bs64lr0.0001seq10edim128ep288.pth"),  #   zwykły Cond
# checkpoint = torch.load(os.path.join(models_dir,"2024-11-06_21-33_bs64lr0.0001seq30edim1024eps100/bs64lr0.0001seq30edim1024ep2drop0.0.pth"),
# checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq30edim1024eps150/bs64lr0.0001seq30edim1024ep150.pth"),  #   top zwykły Cond
# checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq30edim512eps100/bs64lr0.0001seq30edim512ep100.pth"),
# checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq30edim768eps100/bs64lr0.0001seq30edim768ep100.pth"),
# checkpoint = torch.load(os.path.join(models_dir,"2024-11-11_01-54_bs64lr0.0001seq30edim512eps30/bs64lr0.0001seq30edim512ep30drop0.3.pth"),  #   top BiLSTM seq 30
# checkpoint = torch.load(os.path.join(models_dir,"2024-11-11_19-31_bs64lr0.0001seq10edim256eps30/bs64lr0.0001seq10edim256ep30drop0.3.pth"),
# checkpoint = torch.load(os.path.join(models_dir,"2024-11-12_08-37_bs64lr0.0001seq60edim1024eps30/bs64lr0.0001seq60edim1024ep10drop0.0.pth"),    #   seq 60
# checkpoint = torch.load(os.path.join(models_dir,"od_Adama_LSTMcond/2024-11-11_19-48_bs64lr0.0001seq10edim512eps30/bs64lr0.0001seq10edim512ep30drop0.0.pth"),
checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq60edim1024ep20drop0.0.pth"),
                        map_location=torch.device(device) , weights_only=True)
model.load_state_dict(checkpoint)
model.to(device=device)
model.eval()

# criterion = nn.L1Loss(reduction='none').to(device)
criterion = nn.MSELoss(reduction='none').to(device)
# criterion = nn.L1Loss(reduction='none').to(device)

class_val = config['DATASET']['CLASS_VAL']

ov_loss1_arr, ov_loss_arr, ov_loss_true1_arr, ov_loss_true_arr = [], [], [], []


for k, (class_test, thresh) in enumerate(exercises_class_touse.items()):
    ov_loss1_arr, ov_loss_arr, ov_loss_true1_arr, ov_loss_true_arr, df = [], [], [], [], []
    TP_ov, TN_ov, FP_ov, FN_ov = 0, 0, 0, 0
    
    for w_id in TEST_W_IDs:
        # w_id = '01'
        workout = 'w' + w_id
        try:
            test_dataset = ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
                                  skeleton_frame_len=seq_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
                                  sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, 
                                  gender=gender, feature_extractor=FeatureExtractor, transform=Transform, class_test=class_val, plot=True)
        
        except (ValueError, KeyError) as e:
            print(f"Błąd podczas tworzenia test_dataset dla workout '{workout}': {e}")
            continue
        # test_dataset.__getitem__(90)
        shuffle = random.random()
        if shuffle < bad_habits_probability:
            tensor, true_tensor, label_tf = test_dataset.add_bad_habits()
        else:
            tensor, label_tf=test_dataset.true_features()
            true_tensor = tensor
        
        # print(workout, class_test, shuffle < bad_habits_probability)
        
        # Poisson_dataset.__getitem__(25)#85856
        one_hot = test_dataset.one_hot_encode()
        condition = torch.tensor(one_hot, dtype=torch.long)
        step = tensor.shape[0] // seq_len

        loss1_arr, loss_arr, loss_true1_arr, loss_true_arr = [], [], [], []
        classif_pois_arr, classif_true_arr = [], []
        for i in range(0, step*seq_len, seq_len):
        # for i in range(0, poisoned_tensor.shape[0]-seq_len, 1):
            with torch.no_grad():
                p = tensor[i:i+seq_len]
                t = true_tensor[i:i+seq_len]
                c = condition[i:i+seq_len]
                p = p.unsqueeze(0)
                t = t.unsqueeze(0)
                c = c.unsqueeze(0)
                combined_data_tensor = p.to(device)
                true_data_tensor = t.to(device)
                condition_tensor = c.to(device)
                
                outputs = model(combined_data_tensor, condition_tensor)
                outputs_true = model(true_data_tensor, condition_tensor)
                loss = criterion(outputs, combined_data_tensor)
                loss = loss.cpu().numpy()[0, :, :]
                loss_true = criterion(outputs_true, true_data_tensor)
                loss_true = loss_true.cpu().numpy()[0, :, :]
                            
                loss_arr.append(loss)
                loss_true_arr.append(loss_true)
                
                # plt.figure(figsize=(12, 6))
                # plt.plot(loss_true, label = 'label')
                # plt.plot(loss, label = 'label')
                # plt.xlabel('Czas (klatki)')
                # plt.show()
                
        loss_arr = np.concatenate(loss_arr)
        loss_true_arr = np.concatenate(loss_true_arr)
        
        #   Uśrednianie (konwolucja) lossu z sąsiednich klatek
        convolve_kernel = np.array([1/10, 2/10, 4/10, 2/10, 1/10])      #   maska konwolucji
        # convolve_kernel = np.array([1])
        loss_arr = convolve1d(loss_arr, weights=convolve_kernel, axis=0, mode='nearest')
        loss_true_arr = convolve1d(loss_true_arr , weights=convolve_kernel, axis=0, mode='nearest')
        loss_difference_arr = loss_arr - loss_true_arr   
        
        
        #   Pozostawienie tylko największych lossów z wybranej liczby cech
        num_maxindices = 89  #   liczba cech które zostawić, max 89
        loss_maxindices = np.argsort(-loss_arr, axis=1)[:,:num_maxindices]
        loss_arr_maxes = np.take_along_axis(loss_arr, loss_maxindices, axis=1)
        loss_true_maxindices = np.argsort(-loss_true_arr, axis=1)[:,:num_maxindices]
        loss_true_arr_maxes = np.take_along_axis(loss_true_arr, loss_true_maxindices, axis=1)
        loss = np.mean(loss_arr_maxes, axis=1)
        loss_true = np.mean(loss_true_arr_maxes, axis=1)
                                                                                        
        ov_loss_arr.append(loss)
        ov_loss_true_arr.append(loss_true)
        
        #   plot settings
        width_px = 2560
        height_px = 1440
        dpi = 100
        fig_width = width_px / dpi
        fig_height = height_px / dpi
        
        if plot_losses == True:
            #   Zapis plotu 'Distortion score, label, exercise: {ex}' z DatasetExc
            plt.savefig(os.path.join(loss_plots_dir, f'{plt.gca().get_title()}, {workout}.png'), dpi=dpi)
            plt.close()
        else:
            plt.close()
        
        #   Heatmapy lossów w danym ćwiczeniu    ``
        plotting3D.loss_heatmap(loss_arr, f'Heatmap loss, exercise: {class_test}, {workout}', 
                                    plot_show=False, plot_block=False, plot_save=False, plot_dir=loss_plots_dir)
            
        plotting3D.loss_heatmap(loss_true_arr, f'Heatmap loss_true, exercise: {class_test}, {workout}', 
                                    plot_show=False, plot_block=False, plot_save=False, plot_dir=loss_plots_dir)
            
        plotting3D.loss_heatmap(loss_difference_arr, f'Heatmap loss_gained, exercise: {class_test}, {workout}', 
                                    plot_show=False, plot_block=False, plot_save=False, plot_dir=loss_plots_dir)
            
        plotting3D.loss_heatmap(-loss_difference_arr, f'Heatmap loss_lost, exercise: {class_test}, {workout}', 
                                    plot_show=False, plot_block=False, plot_save=False, plot_dir=loss_plots_dir) 
        
        classif_pois = (loss > thresh).astype(int)
        classif_true = (loss_true > thresh).astype(int)
        no_l = step*seq_len
        
        y_pred = np.concatenate([classif_pois, classif_true])
        y_true = np.concatenate([label_tf[0:no_l], np.zeros(no_l)])
        
        # precision = precision_score(y_true, y_pred)
        # # Recall
        # recall = recall_score(y_true, y_pred)
        # #  Accuracy
        # accuracy = accuracy_score(y_true, y_pred)

        TP = np.sum(np.logical_and(y_pred == 1, y_true == 1).astype(int))
        TN = np.sum(np.logical_and(y_pred == 0, y_true == 0).astype(int))
        FP = np.sum(np.logical_and(y_pred == 1, y_true == 0).astype(int))
        FN = np.sum(np.logical_and(y_pred == 0, y_true == 1).astype(int))
        TP_ov += TP
        TN_ov += TN
        FP_ov += FP
        FN_ov += FN
        
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        FPR = FP / (TN + FP)
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        
        print(f'\nExercise: {class_test}, steps: {step}, Loss_bad = {np.mean(loss):.4f}, Loss_true = {np.mean(loss_true):.4f}, '
            f'Loss ratio MSE = {(np.mean(loss)/np.mean(loss_true)):.4f}, Precision = {precision*100:.2f}%, Recall = {recall*100:.2f}%, Accuracy = {accuracy*100:.2f}%')
        
        df_good = pd.DataFrame({'loss': loss_true, 'label': 'Poprawne przykłady'})
        df_bad = pd.DataFrame({'loss': loss, 'label': 'Zatrute przykłady'})

        # Łączenie DataFrame
        df_temp = pd.concat([df_good, df_bad], ignore_index=True)
        df.append(df_temp)
    
    
    # # Wykres KDE z użyciem argumentu 'hue' i 'fill'
    # df = pd.concat(df)
    # sns.kdeplot(data=df, x='loss', hue='label', fill=True)       
    # plt.xlabel('Wartość straty')
    # plt.ylabel('Gęstość')
    # plt.title(f'Loss MSE dla przykładów poprawnych i zatrutych, exercise: {class_test}')
    # plt.show()
        
    ov_loss = np.mean(np.concatenate(ov_loss_arr))
    ov_loss_true = np.mean(np.concatenate(ov_loss_true_arr))

    accuracy = (TP_ov + TN_ov)/(TP_ov + TN_ov + FP_ov + FN_ov)
    
    print(f'\nExercise: {class_test}, steps: {step}, loss = {ov_loss:.4f}, loss_true = {ov_loss_true:.4f}, loss/loss_true = {(ov_loss/ov_loss_true):.4f}, Accuracy = {accuracy*100:.2f}%')