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
from Dataset.DatasetExc import ExcDataset
from Dataset.Transform import Transform


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

TRAIN_W_IDs = ['00', '02', '03', '04', '06', '07', '08', '16', '17', '18']
# VAL_W_IDs = ['14', '15', '19']
VAL_W_IDs = ['01']
TEST_W_IDs = ['01', '05', '12', '13', '20']


#--------------------------------------------------------------------------------------------
#   Training params
#--------------------------------------------------------------------------------------------
skeleton_frame_len = 30
# batch_sizes = (4, 8, 16, 32)
batch_sizes = (512,)
input_dim = 89
# input_dim = 51
seq_len = skeleton_frame_len 
embedding_dims = (768,)
condition_dim = len(excercises_class)
learning_rates = (0.0001,)

current_dir = "/mnt/c/Users/Artur/Documents/Github/mm-fit/mm-fit" # Zmienić ścieżkę pod siebie
# current_dir = "/home/adam/ProjektTrener/ExcerciseNet/mm-fit"
models_dir = "/mnt/c/Users/Artur/Documents/Github/GSN_pracka/Date_Infinite_rep/Trained_models/"
num_epochs = 1 # Ogarnąć liczbę epok najlepszą
workout_in_folder = utility.sorting_folders(current_dir=current_dir)
# workout1 = workout_in_folder[0]
# workout2 = workout_in_folder[1]
class_val = 'all_excercises'

# train_dataset = ExcDataset(workouts_dir=current_dir, workout=workout1, excercises_class=excercises_class, 
#                              skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
#                              sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor , class_test=class_test)

# test_dataset = ExcDataset(workouts_dir=current_dir, workout=workout2, excercises_class=excercises_class, 
#                              skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
#                              sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor , class_test=class_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_list, val_list, test_list = [], [], []
for w_id in TRAIN_W_IDs + VAL_W_IDs + TEST_W_IDs:
    modality_filepaths = {}
    workout = 'w' + w_id
    workout_path = os.path.join(current_dir, 'w' + w_id)
    files = os.listdir(workout_path)
    
    if not files:
        raise Exception('Error: Label file not found for workout {}.'.format(w_id))
    
    # if w_id in TRAIN_W_IDs:
    #     train_list.append(ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
    #                          skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
    #                          sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor,  transform=Transform, class_test=None, plot = False))
    elif w_id in VAL_W_IDs:
        val_list.append(ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
                             skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
                             sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor, transform=Transform, class_test='squats', plot = False))
    # elif w_id in TEST_W_IDs:
    #     test_list.append(ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
    #                          skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
    #                          sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor, transform=Transform, class_test=None, plot = False))
    # else:
    #     raise Exception('Error: Workout {} not assigned to train, test, or val datasets'.format(w_id))

# train_dataset = ConcatDataset(train_list)
val_dataset = ConcatDataset(val_list)
# test_dataset = ConcatDataset(test_list)

# #Poisson_dataset = ExcDataset(workouts_dir=current_dir, workout=workout, excercises_class=excercises_class, 
#                              skeleton_frame_len=skeleton_frame_len, nodes=nodes, limbs=limbs, limbs_angle=limbs_angle,
#                              sym_plane=sym_plane, sym_limbs=sym_limbs, g_nodes=g_nodes, contact_points=contact_points, gender=gender, feature_extractor=FeatureExtractor, transform=Transform, class_test='squats', plot = False)

# train_dataset.__getitem__(85856)#85856

train_labels, val_labels, train_loss_histories, val_loss_histories = [], [], [], []

# Poisson_dataset.one_hot_encode()
# Poisson_dataset.get_noise_data()
# # train_dataset.one_hot_encode()

for i, batch_size in enumerate(batch_sizes):

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
    #                                         sampler=None, pin_memory=True, num_workers=16)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                            sampler=None, pin_memory=True, num_workers=16)

    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
    #                                         sampler=None, pin_memory=True, num_workers=16)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for j, learning_rate in enumerate(learning_rates):
        for k, embedding_dim in enumerate(embedding_dims):

            print(f'Batch size: {batch_size} ({i+1}/{len(batch_sizes)}), Learning rate: {learning_rate} ({j+1}/{len(learning_rates)}), Embedding dimension: {embedding_dim} ({k+1}/{len(embedding_dims)})')
            
            #model = RecurrentAutoencoder(seq_len, input_dim, embedding_dim=embedding_dim)
            models_dir = "/mnt/c/Users/Artur/Documents/Github/GSN_pracka/Date_Infinite_rep/Trained_models/"
            checkpoint = torch.load(os.path.join(models_dir,"bs64lr0.0001seq30edim768eps100/bs64lr0.0001seq30edim768ep100.pth"), weights_only=True)
            model = RecurrentAutoencoderCond(seq_len, input_dim, embedding_dim, condition_dim)
            model.load_state_dict(checkpoint)
            model.eval()
 
            model.to(device=device)

            # model_dir_name = f"bs{batch_size}lr{learning_rate}seq{seq_len}edim{embedding_dim}eps{num_epochs}"
            # model_dir = models_dir + model_dir_name + "/"
            # os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            
            # criterion = nn.CrossEntropyLoss()
            criterion = nn.L1Loss(reduction='mean').to(device)       # L! loss, odporność na outliery, linearna penalizacja błędów
            #criterion = nn.MSELoss(reduction='sum').to(device)    # mean squared error, bardziej penalizuje większe błędy
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Ogarnąć learning rate
            train_loss_history, val_loss_history = [], []

            # Trening

            for epoch in range(num_epochs):
                # model.train()
                # train_losses = []
                # running_loss = 0.0
                counter = 0
                # counter_max = len(train_loader)
                                    
                # for data, condition in val_loader:
                #     counter += 1
                #     tensor = data
            
                    # # Sprawdzenie, czy są dostępne istotne dane
                    # if len(tensor) == 0:
                    #     continue
                    
                    # # Przenoszenie danych na urządzenie
                    # combined_data_tensor = tensor.to(device)
                    # condition_tensor = condition.to(device)
                    
                    # optimizer.zero_grad()
                    
                    # outputs = model(combined_data_tensor, condition_tensor)
                    
                    # loss = criterion(outputs, combined_data_tensor) # Loss wynikający z różnicy rekonstrukcji(outputs) i rzeczywiście zmierzonymi danymi
                    
                    # loss.backward()
                    # optimizer.step()
                    
                    # train_losses.append(loss.item())
                    # running_loss += loss.item()
                    # if counter % 1000 == 0 or counter == counter_max:
                    #     print('Training on batch no. ', counter, '/', counter_max)
                    
                # epoch_loss = running_loss / len(train_loader)
                # train_loss_history.append(epoch_loss)
                
                # print(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {epoch_loss:.4f}')
                
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
                
                # #   Zapisywanie
                # model_name = f"bs{batch_size}lr{learning_rate}seq{seq_len}edim{embedding_dim}ep{epoch+1}.pth"
                # model_path = model_dir + model_name
                # torch.save(model.state_dict(), model_path)
                
                
            # loop_iter = len(learning_rates)*i+j
            
            # train_loss_start = train_loss_history[0]
            # train_loss_end = train_loss_history[num_epochs-1]
            # train_loss_min = np.min(train_loss_history)
            # train_loss_ratio_end = round((((train_loss_end/train_loss_start)-1)*100), 2)
            # train_loss_ratio_max = round((((train_loss_min/train_loss_start)-1)*100), 2)
            # train_loss_min_idx = np.argmin(train_loss_history)
            
            # train_labels.append(f'bs{batch_size}_lr{learning_rate}_edim{embedding_dim}\nLoss change: {train_loss_ratio_end}%\nMax loss change: {train_loss_ratio_max}% on epoch {train_loss_min_idx+1}')
            # train_loss_histories.append(train_loss_history)
            
            # val_loss_start = val_loss_history[0]
            # val_loss_end = val_loss_history[num_epochs-1]
            # val_loss_min = np.min(val_loss_history)
            # val_loss_ratio_end = round((((val_loss_end/val_loss_start)-1)*100), 2)
            # val_loss_ratio_max = round((((val_loss_min/val_loss_start)-1)*100), 2)
            # val_loss_min_idx = np.argmin(val_loss_history)
            
            # val_labels.append(f'bs{batch_size}_lr{learning_rate}_edim{embedding_dim}\nLoss change: {val_loss_ratio_end}%\nMax loss change: {val_loss_ratio_max}% on epoch {val_loss_min_idx+1}')
            # val_loss_histories.append(val_loss_history)
                     
# ## TA CZĘŚĆ TO JEST TESTOWA, WIZUALIZACJA ITP
# plot_folder = "/mnt/c/Users/Artur/Documents/GitHub/GSN_pracka/Date_Infinite_rep/training_plots/"
# plot_name_train = f"sl{seq_len}_epo{num_epochs}_{class_val}_train.png"
# plot_name_test = f"sl{seq_len}_epo{num_epochs}_{class_val}_test.png"
# plot_path_train = plot_folder + plot_name_train
# plot_path_test = plot_folder + plot_name_test
# loss_max_train = np.max(train_loss_histories)
# loss_max_val = np.max(val_loss_histories)
# loss_max = np.max([loss_max_train, loss_max_val])
# #ymax = loss_max * 1.1
# ymax = 1

# plt.figure(figsize=(10, 5))
# for i, train_loss_history in enumerate(train_loss_histories):

#     plt.plot(train_loss_history, label=train_labels[i])

# plt.grid(True, which='both', axis='y', linestyle='--', color='gray')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim(0, ymax)
# yticks = np.arange(0, ymax, 0.05)
# plt.yticks(yticks)
# plt.title(f'Train Loss over Epochs, batch sizes: {batch_sizes}, learning rates: {learning_rates}, embedding dims: {embedding_dims}')
# plt.legend()
# plt.savefig(plot_path_train)
# #plt.show(block=False)

# plt.figure(figsize=(10, 5))
# for i, val_loss_history in enumerate(val_loss_histories):

#     plt.plot(val_loss_history, label=val_labels[i])

# plt.grid(True, which='both', axis='y', linestyle='--', color='gray')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim(0, ymax)
# yticks = np.arange(0, ymax, 0.05)
# plt.yticks(yticks)
# plt.title(f'Test Loss over Epochs, batch sizes: {batch_sizes}, learning rates: {learning_rates}, embedding dims: {embedding_dims}')
# plt.legend()
# plt.savefig(plot_path_test)
# plt.show()
