import numpy as np
import plotting3D
import matplotlib.pyplot as plt
from features_extractor import FeatureExtractor

class Transform():
        
    def __init__(self, nodes, limbs, data3d, limbs_angle, num_points_to_modify=17, mean=0.0, std=0.1) -> None:
        self.nodes = nodes
        self.limbs = limbs
        self.limbs_angle = limbs_angle
        self.num_pts_tm = num_points_to_modify
        self.mean = mean
        self.std = std
        self.feature_extractor = FeatureExtractor(nodes=self.nodes, limbs=self.limbs)
        self.data3d = self.feature_extractor.change_coord_sys(data3d=data3d, change_coord=False)
        self.angles, self.cos_thetas = self.feature_extractor.calc_angle(self.limbs_angle)

    
    def add_noise_to_random_points(self, start_frame=None ):
        """
        Dodaje szum Gaussian do losowo wybranych punktów 3D w chmurze punktów.
        
        Args:
            point_cloud (numpy.ndarray): Tablica punktów 3D o wymiarach (N, 3).
        
        Returns:
            numpy.ndarray: Tablica punktów 3D z dodanym szumem.
        """
        points_noisy = self.data3d.copy()
        pose3d = self.data3d
        total_frames = self.data3d.shape[0]
        max_start_frame = total_frames - 100
    
        # Jeśli mniej niż 100 ramek, nie można zmodyfikować 100 kolejnych
        if max_start_frame < 0:
            raise ValueError("Liczba ramek jest mniejsza niż 100, nie można zmodyfikować 100 kolejnych ramek.")
        
        if start_frame is None:
        # Wybór losowego startowego indeksu ramki do modyfikacji
            start_frame = np.random.randint(0, max_start_frame + 1)
    
        # Generowanie szumu Gaussowskiego
        # noise = np.random.normal(self.mean, self.std, (100, self.data3d.shape[1], self.num_pts_tm))
        noise = np.random.normal(self.mean, self.std, (1, self.data3d.shape[1], self.num_pts_tm))
        x = noise.repeat(100, self.data3d.shape[1])
        # Dodanie szumu do 100 kolejnych ramek, zaczynając od start_frame
        points_noisy[start_frame:start_frame + 100] += noise
        return points_noisy, self.data3d, start_frame
    
    
    def modify_points_for_posture_squats(self, start, end, node_to_mod, ref_angle, mn, mode='hunch', pose_3d = None, angles=None):  
        
        if pose_3d is not None:
            data_3d = pose_3d
        else:
            data_3d = self.data3d
            
        modify_3d = data_3d.copy()
        
        if angles is not None:
            angles = 180 - angles[start:end,:].squeeze(1)
        else:
            angles = 180 - self.angles[start:end, ref_angle]
            
        # gamma = np.clip((angles) / 90, 0.0, 1.0)
        amma = angles / 90
        max = np.max(amma)
        normalize = amma/max
        gamma = np.power(normalize , 2)
        
        for i, node in enumerate(node_to_mod):
            nm = self.nodes[node]
            m = mn[i]
            # Ustalanie kierunku modyfikacji w zależności od trybu
            if mode == 'hunch':
                # Garbienie: Przesunięcie do przodu i w dół
                delta_y = -(gamma * m)  # Stopniowe przesuwanie w osi Y
                delta_z = -(gamma * m)  # Stopniowe przesuwanie w osi Z
                delta_x = np.zeros_like(gamma)
            elif mode == 'arch':
                # Nadmierne wyprostowanie: Przesunięcie do tyłu i w górę
                delta_y = gamma * m  # Przesuwanie w przeciwnym kierunku w osi Y
                delta_z = gamma * m  # Przesuwanie w przeciwnym kierunku w osi Z
                delta_x = np.zeros_like(gamma)
            else:
                raise ValueError("Mode must be either 'hunch' or 'arch'")
            v = np.stack((delta_x, delta_y, delta_z), axis = 1)
            modify_3d[start:end, :, nm]  = data_3d[start:end, :, nm] + v
                   
        bad_3d = self.feature_extractor.change_coord_sys(data3d=modify_3d, change_coord=False)
        bad_angles, bad_cos = self.feature_extractor.calc_angle(self.limbs_angle )
        # end = end-300
        # name = 'squats_false_3d_animation.mp4'
        # plotting3D.animation_frames(name, bad_3d[start:end,:,:], self.limbs, self.limbs_angle, bad_angles[start:end], xx=None, yy=None, zz=None)
        # name1 = 'squats_true_3d_animation.mp4'
        # plotting3D.animation_frames(name1, data_3d[start:end,:,:], self.limbs, self.limbs_angle, self.angles[start:end], xx=None, yy=None, zz=None)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)) 
        # ax1 = fig.add_subplot(121,projection='3d')
        # plotting3D.plot_3d_pose(ax1, bad_3d[start+25,:,:], self.limbs, start+25, self.limbs_angle, bad_angles)
        # ax2 = fig.add_subplot(122,projection='3d')
        # plotting3D.plot_3d_pose(ax2, data_3d[start+25,:,:], self.limbs, start+25, self.limbs_angle, self.angles)   
        # ax1.set_xlim((-1000, 1000)); ax1.set_ylim((-1000, 1000)); ax1.set_zlim((0, 2000))
        # ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        # ax2.set_xlim((-1000, 1000)); ax2.set_ylim((-1000, 1000)); ax2.set_zlim((0, 2000))
        # ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        # plt.tight_layout()  # Adjust layout to prevent overlap
        # plt.show()
        return bad_3d, bad_angles, bad_cos, data_3d
    
    
    def modify_points_for_posture_biceps_curl(self, start, end, node_to_mod, ref_angle1, ref_angle2, mn, mode='hunch', pose_3d = None):  
        
        if pose_3d is not None:
            data_3d = pose_3d
        else:
            data_3d = self.data3d
            
        modify_3d = data_3d.copy()
        r = self.data3d[start:end,:, self.nodes['Right Wrist']] - data_3d[start:end,:, self.nodes['Left Wrist']]
        angles1 = 180 - self.angles[start:end, ref_angle1]
        angles2 = 180 - self.angles[start:end, ref_angle2]
        # gamma1 = np.clip((angles1) / 90, 0.0, 1.0)
        # gamma2 = np.clip((angles2) / 90, 0.0, 1.0)
        amma = angles1 / 90
        max = np.max(amma)
        normalize = amma/max
        gamma1 = np.power(normalize , 2)
        
        amma = angles2 / 90
        max = np.max(amma)
        normalize = amma/max
        gamma2 = np.power(normalize , 2)
        
        gamma = np.where(r[:,2]>0, gamma1, gamma2)
        # gamma = np.power(gamma, 2)
        
        for i, node in enumerate(node_to_mod):
            nm = self.nodes[node]
            m = mn[i]
                    # Ustalanie kierunku modyfikacji w zależności od trybu
            if mode == 'hunch':
                # Garbienie: Przesunięcie do przodu i w dół
                delta_y = -(gamma * m)  # Stopniowe przesuwanie w osi Y
                delta_z = -(gamma * m)  # Stopniowe przesuwanie w osi Z
                delta_x = np.zeros_like(delta_z)
            elif mode == 'arch':
                # Nadmierne wyprostowanie: Przesunięcie do tyłu i w górę
                delta_y = gamma * m  # Przesuwanie w przeciwnym kierunku w osi Y
                delta_z = gamma * m  # Przesuwanie w przeciwnym kierunku w osi Z
                delta_x = np.zeros_like(delta_z)
            else:
                raise ValueError("Mode must be either 'hunch' or 'arch'")
            v = np.stack((delta_x, delta_y, delta_z), axis = 1)
            modify_3d[start:end, :, nm]  = data_3d[start:end, :, nm] + v
            
        bad_3d = self.feature_extractor.change_coord_sys(data3d=modify_3d, change_coord=False)
        bad_angles, bad_cos = self.feature_extractor.calc_angle(self.limbs_angle )
        end = end-300
        # name = 'bicep_curls_false_3d_animation.mp4'
        # plotting3D.animation_frames(name, bad_3d[start:end,:,:], self.limbs, self.limbs_angle, bad_angles[start:end], xx=None, yy=None, zz=None)
        # name1 = 'bicep_curls_true_3d_animation.mp4'
        # plotting3D.animation_frames(name1, data_3d[start:end,:,:], self.limbs, self.limbs_angle, self.angles[start:end], xx=None, yy=None, zz=None)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)) 
        # ax1 = fig.add_subplot(121,projection='3d')
        # plotting3D.plot_3d_pose(ax1, bad_3d[503,:,:], self.limbs, 503, self.limbs_angle, bad_angles) #668
        # ax2 = fig.add_subplot(122,projection='3d')
        # plotting3D.plot_3d_pose(ax2, data_3d[503,:,:], self.limbs, 503, self.limbs_angle, self.angles)   
        # ax1.set_xlim((-1000, 1000)); ax1.set_ylim((-1000, 1000)); ax1.set_zlim((0, 2000))
        # ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        # ax2.set_xlim((-1000, 1000)); ax2.set_ylim((-1000, 1000)); ax2.set_zlim((0, 2000))
        # ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        # plt.tight_layout()  # Adjust layout to prevent overlap
        # plt.show()
        return bad_3d, bad_angles, bad_cos, data_3d
    
    
    def modify_points_for_lunges(self, start, end, node_to_mod, ref_angle1, ref_angle2, mn, pose_3d=None):
        if pose_3d is not None:
            data_3d = pose_3d
        else:
            data_3d = self.data3d

        modify_3d = data_3d.copy()

        angles1 = 180 - self.angles[start:end, ref_angle1]  
        angles2 = 180 - self.angles[start:end, ref_angle2]  

        # gamma_right = np.clip(angles1 / 90, 0.0, 1.0)
        # gamma_left = np.clip(angles2 / 90, 0.0, 1.0)
        
        amma = angles1 / 90
        max = np.max(amma)
        normalize = amma/max
        gamma_right = np.power(normalize , 2)
        
        amma = angles2 / 90
        max = np.max(amma)
        normalize = amma/max
        gamma_left = np.power(normalize , 2)

        pelvis_index = self.nodes.get('Hip')

        pelvis_coords = data_3d[start:end, :, pelvis_index]

        right_knee_coords = data_3d[start:end, :, self.nodes['Right Knee']]
        left_knee_coords = data_3d[start:end, :, self.nodes['Left Knee']]

        dist_right_knee = np.linalg.norm(right_knee_coords - pelvis_coords, axis=1)
        dist_left_knee = np.linalg.norm(left_knee_coords - pelvis_coords, axis=1)

        front_leg = np.where(dist_right_knee > dist_left_knee, 'Right', 'Left')
        back_leg = np.where(front_leg == 'Right', 'Left', 'Right')

        node_leg_map = {
            'Right Knee': 'Right',
            'Right Foot': 'Right',
            'Left Knee': 'Left',
            'Left Foot': 'Left'
        }

        for i, node in enumerate(node_to_mod):
            nm = self.nodes[node]
            m = mn[i]
            node_leg = node_leg_map.get(node)

            if node_leg is None:
                continue

            delta = np.zeros((end - start, 3))

            if node_leg == 'Right':
                is_front_leg = (front_leg == 'Right')
                gamma = np.where(is_front_leg, gamma_right, 1.0 - gamma_left)
            else:
                is_front_leg = (front_leg == 'Left')
                gamma = np.where(is_front_leg, gamma_left, 1.0 - gamma_right)

            direction_vector = data_3d[start:end, :, nm] - pelvis_coords
            norm = np.linalg.norm(direction_vector, axis=1, keepdims=True)
            norm[norm == 0] = 1 
            unit_direction = direction_vector / norm
            delta = -(unit_direction * (gamma[:, np.newaxis] * m))
            modify_3d[start:end, :, nm] = data_3d[start:end, :, nm] + delta

        bad_3d = self.feature_extractor.change_coord_sys(data3d=modify_3d, change_coord=False)
        bad_angles, bad_cos = self.feature_extractor.calc_angle(self.limbs_angle)

        return bad_3d, bad_angles, bad_cos, data_3d
    
    
    def modify_points_for_arch(self, start, end, node_to_mod, ref_angle, mn, pose_3d = None, angles=None):  
        
        if pose_3d is not None:
            data_3d = pose_3d
        else:
            data_3d = self.data3d
            
        modify_3d = data_3d.copy()
        limb_modify = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1])
        angles = 180 - self.angles[start:end,ref_angle]
        # gamma = np.clip((angles) / 90, 0.0, 1.0)
        amma = angles / 90
        max = np.max(amma)
        normalize = amma/max
        gamma = np.power(normalize , 2)
        
        for i,node in enumerate(node_to_mod):
            nm = self.nodes[node]
            m = mn[i]
            if limb_modify [nm] == 1:
                delta_x = -(gamma * m)
                delta_y = np.zeros_like(delta_x)   
                delta_z = gamma * m
                v = np.stack((delta_x, delta_y, delta_z), axis = 1)  
                modify_3d[start:end, :, nm]  = data_3d[start:end, :, nm] + v
            else:
                delta_x = gamma * m
                delta_y = np.zeros_like(delta_x)   
                delta_z = gamma * m
                v = np.stack((delta_x, delta_y, delta_z), axis = 1)  
                modify_3d[start:end, :, nm]  = data_3d[start:end, :, nm] + v
            
        bad_3d = self.feature_extractor.change_coord_sys(data3d=modify_3d, change_coord=False)
        bad_angles, bad_cos = self.feature_extractor.calc_angle(self.limbs_angle )
        # for i in range(bad_3d[start:end].shape[0]):
        # end = end-300
        # name = 'tricep_extensions_false_3d_animation.mp4'
        # plotting3D.animation_frames(name, bad_3d[start:end,:,:], self.limbs, self.limbs_angle, bad_angles[start:end], xx=None, yy=None, zz=None)
        # name1 = 'tricep_extensions_true_3d_animation.mp4'
        # plotting3D.animation_frames(name1, data_3d[start:end,:,:], self.limbs, self.limbs_angle, self.angles[start:end], xx=None, yy=None, zz=None)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)) 
        # ax1 = fig.add_subplot(121,projection='3d')
        # plotting3D.plot_3d_pose(ax1, bad_3d[start+25,:,:], self.limbs, start+25, self.limbs_angle, bad_angles)
        # ax2 = fig.add_subplot(122,projection='3d')
        # plotting3D.plot_3d_pose(ax2, data_3d[start+25,:,:], self.limbs, start+25, self.limbs_angle, self.angles)   
        # ax1.set_xlim((-1000, 1000)); ax1.set_ylim((-1000, 1000)); ax1.set_zlim((0, 2000))
        # ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        # ax2.set_xlim((-1000, 1000)); ax2.set_ylim((-1000, 1000)); ax2.set_zlim((0, 2000))
        # ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        # plt.tight_layout()  
        # plt.show()
        return bad_3d, bad_angles, bad_cos, data_3d
    
    
    def modify_points_for_trans_flank(self, start, end, node_to_mod, ref_angle, div_angle, mn, delta=(0,0,0), mode='inter', pose_3d = None, angles=None):  
        
        if pose_3d is not None:
            data_3d = pose_3d
        else:
            data_3d = self.data3d
        
        if angles is not None:
            angles = 180 - angles[start:end,:].squeeze(1)
        else:
            angles = 180 - self.angles[start:end, ref_angle]
            
        modify_3d = data_3d.copy()
        left_right_limb = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        
        amma = angles / 90
        max = np.max(amma)
        normalize = amma/max
        gamma = np.power(normalize , 2)
        
        if mode == 'inter':
            # gamma = np.clip(np.log(angles) / np.log(div_angle), 0.0, 1.0) 
            # gamma = np.clip((angles) / div_angle, 0.0, 1.0) #180 dumbell_row #90 squats 
            gamma = gamma
        elif mode == 'outer':
            # gamma = -np.clip(np.log(angles) / np.log(div_angle), 0.0, 1.0) 
            # gamma = -np.clip((angles) / div_angle, 0.0, 1.0) #180 dumbell_row #90 squats 
            gamma = -(gamma)
        else:
            raise ValueError("Mode must be either 'inter' or 'outer'")
        
        for i,node in enumerate(node_to_mod):
            nm = self.nodes[node]
            m = mn[i]
            if left_right_limb[nm] == 1:
                if delta[0] == 0:
                    delta_x = np.zeros_like(gamma)  #lateral_shoulder
                else:
                    delta_x = gamma * m    #squats,    dumbell_row
                if delta[1] == 0:
                    delta_y = np.zeros_like(gamma) #squats, lateral_shoulder
                else:
                    delta_y = -(gamma * m)    #dumbell_row, lateral_shoulder
                if delta[2] == 0:
                    delta_z = np.zeros_like(gamma)  #dumbell_row,   squats
                else:
                    delta_z = gamma * m    #lateral_shoulder
                    
                v = np.stack((delta_x, delta_y, delta_z), axis = 1)  
                modify_3d[start:end, :, nm]  = data_3d[start:end, :, nm] + v
            else:
                if delta[0] == 0:
                    delta_x = np.zeros_like(gamma)  #lateral_shoulder
                else:
                    delta_x = -(gamma * m)    #squats,    dumbell_row
                if delta[1] == 0:
                    delta_y = np.zeros_like(gamma) #squats, lateral_shoulder
                else:
                    delta_y = -(gamma * m)   #dumbell_row lateral_shoulder
                if delta[2] == 0:
                    delta_z = np.zeros_like(gamma)  #dumbell_row,   squats
                else:
                    delta_z = gamma * m    #lateral_shoulder

                v = np.stack((delta_x, delta_y, delta_z), axis = 1)  
                modify_3d[start:end, :, nm]  = data_3d[start:end, :, nm] + v
            
        bad_3d = self.feature_extractor.change_coord_sys(data3d=modify_3d, change_coord=False)
        bad_angles, bad_cos = self.feature_extractor.calc_angle(self.limbs_angle )
        # end = end-300
        # name = 'lateral_shoulder_raises_3d_animation.mp4'
        # plotting3D.animation_frames(name, bad_3d[start:end,:,:], self.limbs, self.limbs_angle, bad_angles[start:end], xx=None, yy=None, zz=None, data3d_t=data_3d[start:end,:,:])
        # name1 = 'lateral_shoulder_3d_animation.mp4'
        # plotting3D.animation_frames(name1, data_3d[start:end,:,:], self.limbs, self.limbs_angle, self.angles[start:end], xx=None, yy=None, zz=None)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)) 
        # ax1 = fig.add_subplot(121,projection='3d')
        # plotting3D.plot_3d_pose(ax1, bad_3d[start+25,:,:], self.limbs, start+25, self.limbs_angle, bad_angles)
        # ax2 = fig.add_subplot(122,projection='3d')
        # plotting3D.plot_3d_pose(ax2, self.data3d[start+25,:,:], self.limbs, start+25, self.limbs_angle, self.angles)   
        # ax1.set_xlim((-1000, 1000)); ax1.set_ylim((-1000, 1000)); ax1.set_zlim((0, 2000))
        # ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        # ax2.set_xlim((-1000, 1000)); ax2.set_ylim((-1000, 1000)); ax2.set_zlim((0, 2000))
        # ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        # plt.tight_layout()  # Adjust layout to prevent overlap
        # plt.show()
        return bad_3d, bad_angles, bad_cos, data_3d
    

    def modify_points_for_trans_flick(self, start, end, node_to_mod, limbs_angle, mn, mode='front', pose_3d = None):  
        
        if pose_3d is not None:
            data_3d = pose_3d
        else:
            data_3d = self.data3d
        left_right_limb = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        modify_3d = data_3d.copy()
        angle_deg, cos_thetas = self.feature_extractor.calc_angle(limbs_angle, mode='nodes')
        angles = 180 - angle_deg
        
        amma = angles / 90
        max = np.max(amma)
        normalize = amma/max
        gamma = np.power(normalize , 2)

        if mode == 'front':
            # gamma = -np.clip((angles) / 90, 0.0, 1.0) 
            gamma = -(gamma)
        elif mode == 'back':
            # gamma = np.clip((angles) / 90, 0.0, 1.0) 
            gamma = gamma
        else:
            raise ValueError("Mode must be either 'inter' or 'outer'")
        
        for i, node in enumerate(node_to_mod):
            nm = self.nodes[node]
            m = mn[i]
            delta_x = gamma * m 
            delta_y = np.zeros_like(delta_x) 
            delta_z = np.zeros_like(delta_x)
            v = np.stack((delta_x, delta_y, delta_z), axis = 1).squeeze(2)  
            modify_3d[:, :, nm]  = data_3d[:, :, nm] + v
            
        bad_3d = self.feature_extractor.change_coord_sys(data3d=modify_3d, change_coord=False)
        bad_angles, bad_cos = self.feature_extractor.calc_angle(self.limbs_angle )
        # end = end-300
        # name = 'tricep_extensions_false_3d_animation.mp4'#Zmienic nazwe do zapis
        # plotting3D.animation_frames(name, bad_3d[start:end,:,:], self.limbs, self.limbs_angle, bad_angles[start:end], xx=None, yy=None, zz=None)
        # name1 = 'tricep_extensions_true_3d_animation.mp4' #Zmienic nazwe do zapis
        # plotting3D.animation_frames(name1, data_3d[start:end,:,:], self.limbs, self.limbs_angle, self.angles[start:end], xx=None, yy=None, zz=None)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)) 
        # ax1 = fig.add_subplot(121,projection='3d')
        # plotting3D.plot_3d_pose(ax1, bad_3d[start+50,:,:], self.limbs, start+50, self.limbs_angle, bad_angles)
        # ax2 = fig.add_subplot(122,projection='3d')
        # plotting3D.plot_3d_pose(ax2, data_3d[start+50,:,:], self.limbs, start+50, self.limbs_angle, self.angles)   
        # ax1.set_xlim((-1000, 1000)); ax1.set_ylim((-1000, 1000)); ax1.set_zlim((0, 2000))
        # ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        # ax2.set_xlim((-1000, 1000)); ax2.set_ylim((-1000, 1000)); ax2.set_zlim((0, 2000))
        # ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        # plt.tight_layout()  # Adjust layout to prevent overlap
        # plt.show()
        return bad_3d, bad_angles, bad_cos, data_3d