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
from datetime import datetime


with open('configs/config_Adam.yaml', 'r') as file:
# with open('configs/config_Artur.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Wyświetl dane
# print(config)

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
                  'lateral_shoulder_raises': 8, 'jumping_jacks': 9, 'non_activity': 10} # Typy ćwiczeń

TRAIN_W_IDs = config['DATASET']['TRAIN_W_IDs']
VAL_W_IDs = config['DATASET']['VAL_W_IDs']
TEST_W_IDs = config['DATASET']['TEST_W_IDs']


#--------------------------------------------------------------------------------------------
#   Training params
#--------------------------------------------------------------------------------------------
skeleton_frame_len = config['DATASET']['SEQLEN']
batch_sizes = config['TRAIN']['BATCH_SIZE']
input_dim = config['MODEL']['INPUT_DIM']
seq_len = skeleton_frame_len 
embedding_dims = config['TRAIN']['MOT_DISCR']['EMBEDDING_DIMS']
condition_dim = config['MODEL']['CONDITION_DIM']
learning_rates = config['TRAIN']['MOT_DISCR']['LR']

#current_dir = "/mnt/c/Users/Artur/Documents/Github/mm-fit/mm-fit" # Zmienić ścieżkę pod siebie
current_dir = config['CURRENT_DIR']
models_dir = config['MODELS']
num_epochs = config['TRAIN']['NUM_EPOCHS']
workout_in_folder = utility.sorting_folders(current_dir=current_dir)
class_val = config['DATASET']['CLASS_VAL']
num_workers = config['NUM_WORKERS']

dropout_rates = [0.01]

# dropout_rates = [0.0, 0.03, 0.1, 0.3]
dropout_rates = [0.0, 0.3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_list, val_list, test_list = [], [], []
for w_id in TRAIN_W_IDs + VAL_W_IDs + TEST_W_IDs:
    modality_filepaths = {}
    workout = 'w' + w_id
    workout_path = os.path.join(current_dir, 'w' + w_id)
    files = os.listdir(workout_path)
    
    if not files:
        raise Exception('Error: Label file not found for workout {}.'.format(w_id))
    
    if w_id in TRAIN_W_IDs:
        train_list.append(ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
                             skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
                             sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor,  transform=Transform, class_test=None, plot = False))
    elif w_id in VAL_W_IDs:
        val_list.append(ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
                             skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
                             sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor, transform=Transform, class_test=None, plot = False))
    elif w_id in TEST_W_IDs:
        test_list.append(ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
                             skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
                             sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor, transform=Transform, class_test=None, plot = False))
    else:
        raise Exception('Error: Workout {} not assigned to train, test, or val datasets'.format(w_id))

train_dataset = ConcatDataset(train_list)
val_dataset = ConcatDataset(val_list)
test_dataset = ConcatDataset(test_list)

# #Poisson_dataset = ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
#                              skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
#                              sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor, transform=Transform, class_test='squats', plot = False)

# train_dataset.__getitem__(85856)#85856

train_labels, val_labels, train_loss_histories, val_loss_histories = [], [], [], []
curr_date = datetime.now().strftime("%Y-%m-%d_%H-%M")

for i, batch_size in enumerate(batch_sizes):

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                            sampler=None, pin_memory=True, num_workers=num_workers)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                            sampler=None, pin_memory=True, num_workers=num_workers)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            sampler=None, pin_memory=True, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for j, learning_rate in enumerate(learning_rates):
        for k, embedding_dim in enumerate(embedding_dims):
            for l, dropout_rate in enumerate(dropout_rates):

                print(f'Batch size: {batch_size} ({i+1}/{len(batch_sizes)}), Learning rate: {learning_rate} ({j+1}/{len(learning_rates)}), Embedding dimension: {embedding_dim} ({k+1}/{len(embedding_dims)}), Dropout rate: {dropout_rate} ({l+1}/{len(dropout_rates)})')
                
                # model = RecurrentAutoencoder(seq_len, input_dim, embedding_dim=embedding_dim)
                # model = RecurrentAutoencoderCond(seq_len, input_dim, embedding_dim, condition_dim)
                # model = RecurrentAutoencoderCondBi(seq_len, input_dim, embedding_dim, condition_dim)
                model = RecurrentAutoencoderCond(seq_len, input_dim, embedding_dim, condition_dim)
                model = RecurrentAutoencoderCondBi(seq_len, input_dim, embedding_dim, condition_dim)
                # model = RecurrentAutoencoderCondBiDropout(seq_len, input_dim, embedding_dim, condition_dim, dropout_rate=dropout_rate)
                model.to(device=device)

                model_dir_name = f"bs{batch_size}lr{learning_rate}seq{seq_len}edim{embedding_dim}eps{num_epochs}"
                model_dir = models_dir + f"{curr_date}_" + model_dir_name + "/"
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                
                # criterion = nn.CrossEntropyLoss()
                # criterion = nn.L1Loss(reduction='mean').to(device)       # L! loss, odporność na outliery, linearna penalizacja błędów
                criterion = nn.MSELoss(reduction='mean').to(device)    # mean squared error, bardziej penalizuje większe błędy
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Ogarnąć learning rate
                train_loss_history, val_loss_history = [], []

                # Trening

                for epoch in range(num_epochs):
                    model.train()
                    train_losses = []
                    running_loss = 0.0
                    counter = 0
                    counter_max = len(train_loader)
                                        
                    for data, condition in train_loader:
                        counter += 1
                        tensor = data
                
                        # Sprawdzenie, czy są dostępne istotne dane
                        if len(tensor) == 0:
                            continue
                        
                        # Przenoszenie danych na urządzenie
                        combined_data_tensor = tensor.to(device)
                        condition_tensor = condition.to(device)
                        
                        optimizer.zero_grad()
                        
                        outputs = model(combined_data_tensor, condition_tensor)
                        
                        loss = criterion(outputs, combined_data_tensor) # Loss wynikający z różnicy rekonstrukcji(outputs) i rzeczywiście zmierzonymi danymi
                        
                        loss.backward()
                        optimizer.step()
                        
                        train_losses.append(loss.item())
                        running_loss += loss.item()
                        if counter % 1000 == 0 or counter == counter_max:
                            print('Training on batch no. ', counter, '/', counter_max)
                        
                    epoch_loss = running_loss / len(train_loader)
                    train_loss_history.append(epoch_loss)
                    
                    print(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {epoch_loss:.4f}')
                    
                    # loss na zbiorze testowym
                    counter = 0
                    counter_max = len(val_loader)
                    model.eval()  # Przełączenie modelu w tryb ewaluacji
                    test_running_loss = 0.0
                    with torch.no_grad():  # Wyłączenie gradinetów, nie potrzebne w ewaluacji
                        for data, condition in val_loader:
                            counter += 1
                            tensor = data
                            
                            if len(tensor) == 0:
                                continue
                            
                            combined_data_tensor = tensor.to(device)
                            condition_tensor = condition.to(device)
                            outputs = model(combined_data_tensor, condition_tensor)
                            val_loss = criterion(outputs, combined_data_tensor)
                            test_running_loss += val_loss.item()
                            if counter % 1000 == 0 or counter == counter_max:
                                print('Validation on batch no. ', counter, '/', counter_max)
                        
                    epoch_loss = test_running_loss / len(val_loader)
                    val_loss_history.append(epoch_loss)
                    
                    print(f'Epoch [{epoch+1}/{num_epochs}], Val loss: {epoch_loss:.4f}')
                    
                    # Zapisywanie
                    if (epoch+1) % 5 == 0 or epoch==max(range(num_epochs)) or epoch==0:
                        model_name = f"bs{batch_size}lr{learning_rate}seq{seq_len}edim{embedding_dim}ep{epoch+1}drop{dropout_rate}.pth"
                        model_path = model_dir + model_name
                        torch.save(model.state_dict(), model_path)
                
                
                loop_iter = len(learning_rates)*i+j
                
                train_loss_start = train_loss_history[0]
                train_loss_end = train_loss_history[num_epochs-1]
                train_loss_min = np.min(train_loss_history)
                train_loss_ratio_end = round((((train_loss_end/train_loss_start)-1)*100), 2)
                train_loss_ratio_max = round((((train_loss_min/train_loss_start)-1)*100), 2)
                train_loss_min_idx = np.argmin(train_loss_history)
                
                train_labels.append(f'bs{batch_size}_lr{learning_rate}_edim{embedding_dim}_drop{dropout_rate}, Loss change: {train_loss_ratio_end}%, Max loss change: {train_loss_ratio_max}% on epoch {train_loss_min_idx+1}')
                train_loss_histories.append(train_loss_history)
                
                val_loss_start = val_loss_history[0]
                val_loss_end = val_loss_history[num_epochs-1]
                val_loss_min = np.min(val_loss_history)
                val_loss_ratio_end = round((((val_loss_end/val_loss_start)-1)*100), 2)
                val_loss_ratio_max = round((((val_loss_min/val_loss_start)-1)*100), 2)
                val_loss_min_idx = np.argmin(val_loss_history)
                
                val_labels.append(f'bs{batch_size}_lr{learning_rate}_edim{embedding_dim}_drop{dropout_rate}, Loss change: {val_loss_ratio_end}%, Max loss change: {val_loss_ratio_max}% on epoch {val_loss_min_idx+1}')
                val_loss_histories.append(val_loss_history)
                     
## TA CZĘŚĆ TO JEST TESTOWA, WIZUALIZACJA ITP
plot_folder = config['LOGDIR']
plot_name_train = f"{curr_date}_sl{seq_len}_epo{num_epochs}_all_excercises_train.png"
plot_name_test = f"{curr_date}_sl{seq_len}_epo{num_epochs}_all_excercises_test.png"
plot_path_train = plot_folder + plot_name_train
plot_path_test = plot_folder + plot_name_test
loss_max_train = np.max(train_loss_histories)
loss_max_val = np.max(val_loss_histories)
loss_max = np.max([loss_max_train, loss_max_val])
#ymax = loss_max * 1.1
ymax = 0.3

plt.figure(figsize=(10, 5))
for i, train_loss_history in enumerate(train_loss_histories):

    plt.plot(train_loss_history, label=train_labels[i])

plt.grid(True, which='both', axis='y', linestyle='--', color='gray')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, ymax)
yticks = np.arange(0, ymax, 0.05)
plt.yticks(yticks)
plt.title(f'Train Loss over Epochs, batch sizes: {batch_sizes}, learning rates: {learning_rates}, embedding dims: {embedding_dims}')
plt.legend()
plt.savefig(plot_path_train)
#plt.show(block=False)

plt.figure(figsize=(10, 5))
for i, val_loss_history in enumerate(val_loss_histories):

    plt.plot(val_loss_history, label=val_labels[i])

plt.grid(True, which='both', axis='y', linestyle='--', color='gray')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, ymax)
yticks = np.arange(0, ymax, 0.05)
plt.yticks(yticks)
plt.title(f'Test Loss over Epochs, batch sizes: {batch_sizes}, learning rates: {learning_rates}, embedding dims: {embedding_dims}')
plt.legend()
plt.savefig(plot_path_test)
plt.show()
