import os
import random
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import utility
import plotting3D
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


class ExcDataset(data.Dataset):

  def __init__(self, workouts_dir, excercises_class, workout, skeleton_frame_len, nodes, limbs, 
               limbs_angle, sym_plane, sym_limbs, g_nodes, contact_points, gender, feature_extractor, transform, class_test = None, plot = True):
      self.workouts_dir = workouts_dir
      self.excercises_class = excercises_class
      self.workout = workout
      self.frame_count = 0
      self.skeleton_frame_len = skeleton_frame_len
      self.class_test = class_test
      self.contact_points = contact_points
      self.g_nodes = g_nodes
      self.gender = gender
      self.label = self._get_label()
      if self.class_test not in self.label and self.class_test is not None:
        raise KeyError(f"Brak ćwiczenia dla workoutu {self.workout}, {self.class_test}")
      self.pose3D = self._get_pose3d()
      self.nodes = nodes
      self.limbs = limbs
      self.limbs_angle = limbs_angle
      self.sym_plane = sym_plane
      self.sym_limbs = sym_limbs
      self.one_hot_encodes = self.one_hot_encode()
      self.feature_extractor = feature_extractor(nodes=self.nodes, limbs=self.limbs)
      self.plot = plot
      self.transform = transform
      # self.stride = self.skeleton_frame_len // 2
    
  def __len__(self):
      return self.pose3D.shape[1] - self.skeleton_frame_len + 1

  
  def __getitem__(self, idx):
    frame = self.pose3D[0, idx, 0]
    data_3d, one_hot_vector = self._strip_data(frame, idx)
    combined_data_tensor = self._get_feature(data_3d, frame)
  
    condition = torch.tensor(one_hot_vector, dtype=torch.long)
    return combined_data_tensor, condition 
  
  
  def _get_feature(self, data_3d, frame, change_coord = True):
    homo1 = self.feature_extractor.change_coord_sys(data3d=data_3d, change_coord=change_coord)
    angles, cos_thetas = self.feature_extractor.calc_angle(self.limbs_angle)
    zz, xx1, yy1, normal1, D1= self.feature_extractor.define_plane(sym_plane=self.sym_plane)
    r_norm, r, p1, p2, p3, p4, v1, v2 = self.feature_extractor.is_symmetric(self.sym_limbs, normal1, D1, xx1, yy1, zz)
    trajectory, displacement_v = self.feature_extractor.compute_trajectory()
    g_angle, g_degree, cent_mass, cont_points = self.feature_extractor.gravity_angle(self.g_nodes, self.contact_points, self.gender)
        
    if trajectory.shape[0] < self.skeleton_frame_len:
        trajectory = np.pad(trajectory, ((0, self.skeleton_frame_len - trajectory.shape[0]), (0, 0), (0, 0)), mode='constant')
        displacement_v = np.pad(displacement_v, ((0, self.skeleton_frame_len - displacement_v.shape[0]), (0, 0), (0, 0)), mode='constant')
    else:
        trajectory = trajectory[:self.skeleton_frame_len, :, :]
        displacement_v = displacement_v[:self.skeleton_frame_len, :, :]
        
    # Wyświetlanie cech aby sprawdzić czy są poprawne
    if self.plot == True:
      for f in range(self.skeleton_frame_len):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111,projection='3d')
        plotting3D.plot_3d_pose(ax, homo1[f,:,:], self.limbs, f, angles)
        #plotting3D.visualize_sym_vectors(r, p1, p2, p3, p4, v1, v2, f, D1, normal1, xx1, yy1, zz, ax)
        #plotting3D.plot_g_angle(ax, homo1, self.g_nodes, g_degree, f, cent_mass, cont_points)
        # plotting3D.plot_trajectory(ax, homo1, displacement_v, f)
        plt.ion() 
        ax.set_xlim((-1000, 1500))
        ax.set_ylim((-1000, 1000))
        ax.set_zlim((0, 2000))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(str(frame + f))
        plt.show()
    # Równoczesne przekształcanie wszystkich tablic do (max_timesteps, features)
    r_norm_reshaped = r_norm.reshape((self.skeleton_frame_len, -1))
    trajectory_reshaped = trajectory.reshape((self.skeleton_frame_len, -1))
    g_angles_reshaped = g_angle.reshape((self.skeleton_frame_len, -1))
    combined_data = np.concatenate((r_norm_reshaped, cos_thetas, trajectory_reshaped, g_angles_reshaped), axis=1)

    combined_data_tensor = torch.tensor(combined_data, dtype=torch.float32)
    
    return combined_data_tensor 
  
  
  def _strip_data(self, frame, idx):
    one_hot_vector = self.one_hot_encodes[idx: idx+self.skeleton_frame_len, :]
    data_3d = self.pose3D[:, idx: idx+self.skeleton_frame_len,:]
    data_3d = np.swapaxes(data_3d, 0, 1)
    return data_3d[:,:,1:], one_hot_vector
  
  
  def one_hot_encode(self):
    label = 'non_activity'
    one_hot = np.zeros((self.pose3D.shape[1], len(self.excercises_class)), dtype=int)
    one_hot[:, self.excercises_class[label]] = 1
    for ex in self.label:
      for f in self.label[ex]: 
        start = int(f[0])
        end = int(f[1])
        mask = (self.pose3D[0, :, 0] >= start) & (self.pose3D[0, :, 0] <= end)
            
        one_hot[mask, :] = 0  
        one_hot[mask, self.excercises_class[ex]] = 1
    
    return one_hot
  
  
  def get_noise_data(self):
    data_3d = np.swapaxes(self.pose3D, 0, 1)
    self.skeleton_frame_len = data_3d.shape[0]
    transform = self.transform(self.nodes, self.limbs, data_3d, num_points_to_modify=17, mean=0, std=30)
    noise_data3d, pose3d, start_frame = transform.add_noise_to_random_points(start_frame=1)
    data_tensor_true = self._get_feature(pose3d, frame = 0)
    poisoned_tensor= self._get_feature(noise_data3d, frame = 0)
    
    return noise_data3d, pose3d, data_tensor_true, poisoned_tensor
  
  
  def add_bad_habits(self): 
    # self.pose3D = self.average_neighbors(self.pose3D, window_size = 5)    # już uśredniamy w _get_pose3d
    data_3d = np.swapaxes(self.pose3D, 0, 1)[:,:,1:]
    self.skeleton_frame_len = data_3d.shape[0]
    self.plot = False
    data_3d = self.feature_extractor.change_coord_sys(data3d=data_3d, change_coord=True)
    true_3d = data_3d.copy()
    angles_t, cos_thetas = self.feature_extractor.calc_angle(self.limbs_angle)
    transform = self.transform(self.nodes, self.limbs, data_3d, self.limbs_angle, num_points_to_modify=17, mean=0, std=10)
    label_tf = np.zeros(data_3d.shape[0])
    
    for ex in self.label:
      i=0
      for f in self.label[ex]: 
        st= int(f[0])
        en = int(f[1])
        start = int(np.where(self.pose3D[0,:,0] == st)[0])
        end = int(np.where(self.pose3D[0,:,0] == en)[0])
        i=i+1
        
        if ex == 'squats':
          node_to_modify = ['Thorax', 'Neck/Nose', 'Head','Right Elbow','Right Shoulder',
                            'Right Wrist','Left Shoulder',
                            'Left Elbow','Left Wrist']
          # mn = [50,50,50,50,50,50,50,50,50]
          mn = [150,150,150,150,150,150,150,150,150]
          bad_3d, bad_angles, bad_cos, data3d  = transform.modify_points_for_posture_squats(start, end, node_to_modify, ref_angle=9, mn =mn, mode='hunch', pose_3d=data_3d)
          node_to_modify = ['Left Knee','Right Knee']
          mn = [150,150]
          bad_3d, bad_angles, bad_cos, data3d = transform.modify_points_for_trans_flank(start, end, node_to_modify, ref_angle=9, div_angle=90, delta=(1,0,0), mn=mn,  mode='inter', pose_3d=bad_3d)
          end1 = 120
          
          name = f'squats{i}_animation.mp4'
          plotting3D.plot_3d_pose( bad_3d[start+25,:,:], self.limbs, start+25, self.limbs_angle, bad_angles)
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:])       
          distort_thresh = 300

        elif ex == 'pushups':
          r = None
        
        elif ex == 'dumbbell_shoulder_press': 
          node_to_modify = ['Right Elbow','Right Wrist',
                            'Left Elbow','Left Wrist']
          # mn = [10,10,10,10]
          mn = [50,100,50,100]
          #node_to_modify = ['Right Wrist','Left Wrist']
          bad_3d, bad_angles, bad_cos, data3d  = transform.modify_points_for_arch(start, end, node_to_modify, ref_angle=3, mn=mn, pose_3d=data_3d)
          end1 = 120
          name = f'dumbbell_shoulder_press{i}_animation.mp4'
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:])
          distort_thresh = 200

        elif ex == 'lunges': 
          node_to_modify = ['Right Foot','Right Knee',
                            'Left Foot','Left Knee']
          mn = [150,150,150,150]
          bad_3d, bad_angles, bad_cos, data3d  = transform.modify_points_for_lunges(start, end, node_to_modify, ref_angle1=9, ref_angle2 = 10, mn=mn, pose_3d=data_3d)
          node_to_modify = ['Left Knee','Right Knee']
          mn=[150,150]
          bad_3d, bad_angles, bad_cos, data3d = transform.modify_points_for_trans_flank(start, end, node_to_modify, ref_angle=9, div_angle=90, delta=(1,0,0), mn=mn,  mode='inter', pose_3d=bad_3d)
          node_to_modify = ['Thorax', 'Neck/Nose', 'Head','Right Elbow','Right Shoulder',
                            'Right Wrist','Left Shoulder',
                            'Left Elbow','Left Wrist']
          # mn = [10,10,10,10,10,10,10,10,10]
          mn = [150,150,150,150,150,150,150,150,150]
          bad_3d, bad_angles, bad_cos, data3d  = transform.modify_points_for_posture_squats(start, end, node_to_modify, ref_angle=9, mn =mn, mode='hunch', pose_3d=bad_3d)
          end1 = 120
          name = f'lunges{i}_animation.mp4'
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:])
          distort_thresh = 300

        elif ex == 'dumbbell_rows':
          node_to_modify = ['Right Elbow', 'Right Wrist', 'Left Elbow', 'Left Wrist'] 
          mn = [150,500,150,500]
          bad_3d, bad_angles, bad_cos, data3d = transform.modify_points_for_trans_flank(start, end, node_to_modify, ref_angle=3, div_angle=180, delta=(1,1,0), mn=mn,  mode='outer', pose_3d=data_3d)
          end1 = 120
          name = f'dumbbell_rows{i}_animation.mp4'
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:])
          distort_thresh = 400

        elif ex == 'situps': 
          limbs_angle = [(5,0,7),]
          angle_deg, cos_thetas = self.feature_extractor.calc_angle(limbs_angle, mode='nodes')
          node_to_modify = ['Thorax', 'Neck/Nose', 'Head','Right Elbow','Right Shoulder',
                            'Right Wrist','Left Shoulder',
                            'Left Elbow','Left Wrist']
          mn = [150,150,150,150.150,150,150,150,150,150]
          bad_3d, bad_angles, bad_cos, data3d  = transform.modify_points_for_trans_flick(start, end, node_to_modify, limbs_angle, mn=mn, mode='front', pose_3d=data_3d)
          mn=[200,200]
          node_to_modify = ['Right Elbow', 'Left Elbow']
          bad_3d, bad_angles, bad_cos, data3d = transform.modify_points_for_trans_flank(start, end, node_to_modify, ref_angle=9, div_angle=180, delta=(0,1,0), mn=mn,  mode='inter', pose_3d=bad_3d, angles = angle_deg)
          end1 = 120
          name = f'situps{i}_animation.mp4'
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:])
          distort_thresh = 700

        elif ex == 'tricep_extensions':
          # node_to_modify = ['Right Elbow', 'Right Wrist', 'Left Elbow', 'Left Wrist']  
          node_to_modify = ['Right Wrist', 'Left Wrist'] 
          mn = [150,150]
          bad_3d, bad_angles, bad_cos, data3d  = transform.modify_points_for_arch(start, end, node_to_modify, ref_angle=1, mn=mn, pose_3d=data_3d)
          end1 = 120
          name = f'tricep_extensions{i}_animation.mp4'
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:])
          distort_thresh = 200

        elif ex == 'bicep_curls': 
          node_to_modify = ['Thorax', 'Neck/Nose', 'Head','Right Elbow','Right Shoulder',
                            'Right Wrist','Left Shoulder',
                            'Left Elbow','Left Wrist']
          mn = [150,150,150,150,150,150,150,150,150]

          bad_3d, bad_angles, bad_cos, data3d  = transform.modify_points_for_posture_biceps_curl(start, end, node_to_modify, ref_angle1=0, ref_angle2=1, mn=mn, mode='arch', pose_3d=data_3d)
          end1 = 120
          name = f'bicep_curls{i}_animation.mp4'
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:])
          distort_thresh = 350

        elif ex == 'lateral_shoulder_raises': 
          node_to_modify = ['Right Elbow', 'Right Wrist', 'Left Elbow', 'Left Wrist'] 
          # mn = [25,25,25,25]
          mn = [50,50,50,50]
          bad_3d, bad_angles, bad_cos, data3d = transform.modify_points_for_trans_flank(start, end, node_to_modify, ref_angle=3, div_angle=180, delta=(0,0,1), mn=mn, mode='inter', pose_3d=data_3d)
          
          node_to_modify = ['Right Wrist', 'Left Wrist'] 
          # mn = [50, 50]
          mn = [150,150]
          limbs_angle = [(11,8,7),]
          angle_deg, cos_thetas = self.feature_extractor.calc_angle(limbs_angle, mode='nodes')
          bad_3d, bad_angles, bad_cos, data3d = transform.modify_points_for_trans_flank(start, end, node_to_modify, ref_angle=8, div_angle=90, delta=(1,1,0), mn=mn, mode='inter', pose_3d=bad_3d, angles=angle_deg)
          
          node_to_modify = ['Thorax', 'Neck/Nose', 'Head','Right Elbow','Right Shoulder',
                            'Right Wrist','Left Shoulder',
                            'Left Elbow','Left Wrist']
          # mn = [50,50,50,50,50,50,50,50,50]
          mn = [175,175,175,175,175,175,175,175,175]
          bad_3d, bad_angles, bad_cos, data3d  = transform.modify_points_for_posture_squats(start, end, node_to_modify, ref_angle=9, mn =mn, mode='arch', pose_3d=bad_3d, angles=angle_deg)
          end1 = 120
          name = f'lateral_shoulder_raises{i}_animation.mp4'
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:])
          distort_thresh = 860

        elif ex == 'jumping_jacks':
          node_to_modify = ['Thorax', 'Neck/Nose', 'Head','Right Elbow','Right Shoulder',
                            'Right Wrist','Left Shoulder',
                            'Left Elbow','Left Wrist']
          # mn = [50,50,50,50,50,50,50,50,50]
          mn = [100,100,100,100,100,100,100,100,100]
          bad_3d, bad_angles, bad_cos, data3d = transform.modify_points_for_posture_squats(start, end, node_to_modify, ref_angle=3, mn=mn, mode='hunch', pose_3d=data_3d)
          node_to_modify = ['Left Knee','Right Knee']
          # mn = [25,25]
          mn = [50,50]
          bad_3d, bad_angles, bad_cos, data3d = transform.modify_points_for_trans_flank(start, end, node_to_modify, ref_angle=6, div_angle=180, delta=(1,0,0), mn=mn,  mode='inter', pose_3d=bad_3d)
          end1 = 120
          name = f'jumping_jacks{i}_animation.mp4'
          plotting3D.animation_frames(name, bad_3d[start:end1,:,:], self.limbs, self.limbs_angle, bad_angles[start:end1], xx=None, yy=None, zz=None, data3d_t=true_3d[start:end1,:,:], angles_t=angles_t[start:end1,:]) 
          distort_thresh = 200            
        
        data_3d = bad_3d

        # distort_thresh = 200  
        # distorts = data_3d - true_3d
        # distorts = np.linalg.norm(distorts, axis=1)
            
        # distorts_nonzero = np.where(distorts != 0, distorts, np.nan)
        # distorts_score = np.nanmean(distorts_nonzero, axis=1) * (1 + (np.count_nonzero(distorts[0, :]) * 0.3))
        # label_tf[start:end] = np.where(distorts_score[start:end] <= distort_thresh, label_tf[start:end], 1)
            
        # plt.figure(figsize=(12, 6))
        # plt.plot(label_tf, label = 'label')
        # plt.plot(distorts_score/distort_thresh, label = 'distortion score')
        # plt.xlabel('Czas (klatki)')
        # plt.show   

        # print('')
        
        #   Wykres liniowy
        # plt.figure(figsize=(12, 6))
        # for i in range(distortions.shape[1]):
        #     plt.plot(distortions[:, i], label=f'Keypoint {i+1}')
        # plt.xlabel('Czas (klatki)')
        # plt.ylabel('Odkształcenie')
        # plt.title('Wykres liniowy odkształceń keypointów w czasie: {ex}_{i}')
        # plt.legend(loc='upper right', ncol=2)
        # plt.tight_layout()
        # plt.show()
          
        #   Wykres 3D
        # X = np.arange(distortions.shape[0])
        # Y = np.arange(distortions.shape[1])
        # X, Y = np.meshgrid(X, Y)
        # Z = distortions.T  # Transponujemy, aby dopasować wymiary
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, Z, cmap='viridis')
        # ax.set_xlabel('Czas (klatki)')
        # ax.set_ylabel('Keypointy')
        # ax.set_zlabel('Odkształcenie')
        # ax.set_title('Wykres 3D odkształceń keypointów w czasie: {ex}_{i}')
        # plt.show()
          
        #   Wykres heatmap
        # plt.figure(figsize=(10, 6))
        # plt.imshow(distorts.T, aspect='auto', cmap='viridis', interpolation='nearest')
        # plt.colorbar(label='Odkształcenie')
        # plt.xlabel('Czas (klatki)')
        # plt.ylabel('Keypointy')
        # plt.yticks(range(distorts.shape[1]), [f'KP{i+1}' for i in range(distorts.shape[1])])
        # plt.title(f'Mapa ciepła odkształceń keypointów w czasie: {ex}_{i}')
        # plt.tight_layout()
        # plt.show()
                 
    distorts = data_3d - true_3d
    distorts = np.linalg.norm(distorts, axis=1)
        
    distorts_nonzero = np.where(distorts != 0, distorts, np.nan)
    nan_rows = np.all(np.isnan(distorts_nonzero), axis=1)
    distorts_nonzero[nan_rows, :] = 0
    
    distorts_score = np.nanmean(distorts_nonzero, axis=1) * (1 + (np.count_nonzero(distorts[0, :]) * 0.3))
    label_tf = np.where(distorts_score >= distort_thresh, 1, 0)
    
    width_px = 2560
    height_px = 1440
    dpi = 100
    fig_width = width_px / dpi
    fig_height = height_px / dpi
        
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(label_tf, label = 'label')
    plt.plot(distorts_score/distort_thresh, label = 'distortion score')
    plt.xlabel('Czas (klatki)')
    plt.ylabel('Distortion score, label')
    plt.title(f'Distortion score, label, exercise: {ex}')
    # plt.show(block=True) 
     
    features_false = self._get_feature(data_3d, 0, False)
    features_true = self._get_feature(true_3d, 0, False)
    
    return features_false, features_true, label_tf
  
  
  def true_features(self):
    self.plot = False
    true_3d = np.swapaxes(self.pose3D, 0, 1)
    label_tf = np.zeros(true_3d.shape[0])
    self.skeleton_frame_len = true_3d.shape[0]
    features_true = self._get_feature(true_3d[:,:,1:], 0, True)
    # self.plot = False
    return features_true, label_tf
  
  
  def _get_pose3d(self):
    file_path_3d = utility.create_fp_pose3D(self.workouts_dir, self.workout)
    pose3d = utility.load_pose3d(file_path_3d)
    pose3d = self.average_neighbors(pose3d, window_size = 5)    #   UŚREDNIANIE SĄSIADÓW 
    ##Jeżeli chcemy tylko jedną klase ćwiczeń zbadać dla danego workoutu
    if self.class_test is not None:
      class_label={}
      class_label[self.class_test] = self.label[self.class_test]
      self.label = class_label
      pose3d = self._strip_class(pose3d)
    pose3d = self.average_neighbors(pose3d, window_size = 5)    #   UŚREDNIANIE SĄSIADÓW 
    return pose3d
  

  def _get_label(self):
      label = utility.sign_label_to_workout(workout=self.workout, current_dir=self.workouts_dir)
      return label
  
  
  ## Przycięcię data3d jeżeli chcemy tylko 1 klase dla danego workout'u
  def _strip_class(self, pose3d):
    data_list = []
    for f in self.label[self.class_test]: 
      start = int(f[0])
      end = int(f[1])
      data_class = pose3d[:, np.where(((pose3d[0, :, 0]) >= start) & ((pose3d[0, :, 0]) <= end))[0], :]
      data_list.append(data_class)
    pose3D = np.concatenate(data_list, axis=1)
    return pose3D   
  
  
  
  # def average_neighbors(self, pose3D, window_size=5):
  #   num_layers, num_rows, num_columns = pose3D.shape
  #   avg_pose3D = np.zeros((num_layers, num_rows - window_size + 1, num_columns))
  #   kernel = math.floor(window_size/2)

  #   for layer in range(num_layers):
  #       for i in range(kernel, num_rows - kernel):
  #           avg_pose3D[layer, i - kernel] = np.mean(pose3D[layer, i - kernel:i + kernel + 1], axis=0)
    
  #   pose3D = avg_pose3D
  #   return pose3D
 
  def average_neighbors(self, pose3D, window_size=5):
      num_layers, num_rows, num_columns = pose3D.shape
      avg_pose3D = np.zeros((num_layers, num_rows, num_columns), dtype=float)
      avg_pose3D[:, :, 0] = pose3D[:, :, 0]
      kernel = window_size // 2

      for layer in range(num_layers):
          for i in range(num_rows):
              start = max(0, i - kernel)
              end = min(num_rows, i + kernel + 1)
              avg_pose3D[layer, i, 1:] = np.mean(pose3D[layer, start:end, 1:], axis=0)
      
      return avg_pose3D
