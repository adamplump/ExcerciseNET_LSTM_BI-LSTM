import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import plotting3D
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, paired_cosine_distances, paired_euclidean_distances


class FeatureExtractor():
    
    def __init__(self, nodes, limbs) -> None:
        self.nodes = nodes
        self.limbs = limbs
        # self.data3d = data3d
        
    
    def change_coord_sys(self, data3d, change_coord = True):
        #Przesunięcię układu współrzędnych do punktu wyznaczonego jako środek między stopami. Obrót punktów o dany kąt aby "wyprostować postać"
        self.data3d = data3d
        if change_coord == True:
            p1 = self.data3d[:,:,self.nodes['Right Foot']]
            p2 = self.data3d[:,:,self.nodes['Left Foot']]
            p3 = self.data3d[:,:,self.nodes['Right Hip']]
            p4 = self.data3d[:,:,self.nodes['Left Hip']]
            p5 = self.data3d[:,:,self.nodes['Right Knee']]
            ps = (p1 + p2)/2
            
            n= np.repeat(ps[:, :, np.newaxis], 17, axis=2)
            r = self.data3d -n
            p3 = r[:,:,self.nodes['Right Hip']]
            p5 = r[:,:,self.nodes['Right Knee']]
            vectors_v = p3 - p5
            
            norms_v = np.linalg.norm(vectors_v, axis=1, keepdims=True)
            vectors_v_norm = vectors_v / norms_v
            y_axis = np.array([0, 1, 0])

            # Rozszerzanie wektora osi Z do tego samego kształtu co vectors_v_norm
            y_axis_expanded = np.tile(y_axis, (vectors_v_norm.shape[0], 1))
            
            # Obliczanie osi obrotu (iloczyn wektorowy)
            rotation_axes = np.cross(y_axis_expanded, vectors_v_norm)

            # Obliczanie norm osi obrotu, aby uniknąć podziału przez zero
            norms_rotation_axes = np.linalg.norm(rotation_axes, axis=1, keepdims=True)

            # Normalizacja osi obrotu, aby uzyskać jednostkowe wektory
            rotation_axes_normalized = np.where(norms_rotation_axes != 0, rotation_axes / norms_rotation_axes, rotation_axes)

            # Obliczanie kąta obrotu (iloczyn skalarny)
            cos_thetas = np.einsum('ij,ij->i', y_axis_expanded, vectors_v_norm)
            angles_rad = -(np.arccos(cos_thetas)) 
            #Docelowo mozna na stałe dać kąt o ile ciało jest źle obrócone np.float64(-0.4113759260012736) w rad
            angle_deg = np.degrees(angles_rad)
            # W deg: np.float64(-23.570104353159042) - kąt między biodrem a układem 
            
            # Macierz rotacji za pomocą kąta i osi obrotu dla każdej ramki
            rotation_matrices = R.from_rotvec(angles_rad[:, np.newaxis] * rotation_axes_normalized).as_matrix()
            # Macierz rotacji: rotation_matrices[0]
            #     array([[ 0.97688147,  0.21049542,  0.03734003],
            #    [-0.21049542,  0.9165715 ,  0.33998289],
            #    [ 0.03734003, -0.33998289,  0.93969003]])
        
            rotation_mat= np.array([[ 0.97688147,  0.21049542,  0.03734003],
                                    [-0.21049542,  0.9165715 ,  0.33998289],
                                    [ 0.03734003, -0.33998289,  0.93969003]])
            # Przekształcenie punktów za pomocą macierzy rotacji dla każdej ramki
            points_rotated = np.einsum('ijk,ikl->ijl', rotation_matrices, r)
            points_rotated3 = np.einsum('jk,ikl->ijl', rotation_mat, r)

            self.data3d = points_rotated3
        else:
            points_rotated3 = self.data3d
        
        return points_rotated3[:,:,:]    
        
        
    def calc_angle(self, joints_list, mode='nodes'):
        # Oblicza kąt ABC dla 3 dowolnych punktów w przestrzeni
        # joints_list : N x 3 ('nodes') lub F x 3 x 3 ('absolute'), gdzie N to liczba stawów
        #  mode = 'nodes' lub 'absolute' - nodes używa keypointów z data3d, absolute wsółrz. punktów
        # A_index, B_index, C_index - numery keypointów które bierzemy
        valid_modes = ['nodes', 'absolute']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
        
        valid_modes = ['nodes', 'absolute']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
        
        valid_modes = ['nodes', 'absolute']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
        
        angle_list = []
        cos_theta_list = []
        cos_li = []
        sin_li = []
        BC_li = []
        BA_li = []
        
        for _, joint in enumerate(joints_list):
            if mode == 'nodes':
                A = self.data3d[:, :, joint[0]]
                B = self.data3d[:, :, joint[1]]
                C = self.data3d[:, :, joint[2]]
            elif mode == 'absolute':
                A = joint[:,:,0]
                B = joint[:,:,1]
                C = joint[:,:,2]
        
        for _, joint in enumerate(joints_list):
            if mode == 'nodes':
                A = self.data3d[:, :, joint[0]]
                B = self.data3d[:, :, joint[1]]
                C = self.data3d[:, :, joint[2]]
            elif mode == 'absolute':
                A = joint[:,:,0]
                B = joint[:,:,1]
                C = joint[:,:,2]

            BA = A-B
            BC = C-B
            BA = A-B
            BC = C-B

            dot_product = np.einsum('ij,ij->i', BA, BC)

            norm_BA = np.linalg.norm(BA, axis=1)
            norm_BC = np.linalg.norm(BC, axis=1)
            BA_norm = BA/norm_BA[:,np.newaxis]
            BC_norm = BC/norm_BC[:,np.newaxis]
            BA_li.extend(BA_norm)
            BC_li.extend(BC_norm)

            cos_theta = dot_product / (norm_BA * norm_BC)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            theta_radians = np.arccos(cos_theta)
            theta_degrees = np.degrees(theta_radians)
            cos_theta_list.extend(cos_theta)
            angle_list.extend(theta_degrees)
            
            cos_angles = np.cos(theta_radians)
            sin_angles = np.sin(theta_radians)
            cos_li.extend(cos_angles)
            sin_li.extend(sin_angles)
            
        angles_unit_vectors = np.stack((cos_li, sin_li))  # (10, 11, 2)
        angles_unit_vectors = angles_unit_vectors.reshape(self.data3d.shape[0], -1) 
        limb_vectors = np.stack((BA_li, BC_li), axis = 0)
        limb_vectors_reshaped = limb_vectors.reshape(self.data3d.shape[0], -1)
        combined_data = np.concatenate((angles_unit_vectors, limb_vectors_reshaped), axis=1)
        
        thetas_degrees = np.array(angle_list).T
        thetas_degrees = thetas_degrees.reshape((self.data3d.shape[0], len(joints_list)), order='F')
        thetas_degrees = thetas_degrees.reshape((self.data3d.shape[0], len(joints_list)), order='F')
        cos_thetas = np.array(cos_theta_list).T
        cos_thetas = cos_thetas.reshape((self.data3d.shape[0], len(joints_list)), order='F')

        return thetas_degrees, cos_thetas
    
    
    def define_plane(self, sym_plane):
        # Definiuje płaszczyznę przez trzy punkty. - dodana wersja dla 
        # większej libczy klatek jak i również dla 1 
        # Wektory w płaszczyźnie
        p1 = self.data3d[:, :, sym_plane[0][0]]
        p2 = self.data3d[:, :, sym_plane[0][1]]
        p3 = self.data3d[:, :, sym_plane[0][2]]
        v1 = p2 - p1
        v2 = p3 - p1
        p1_ = p1
        p2_ = p2
        p3_ = p3
        normal1 = np.cross(v1, v2)
        n = np.linalg.norm(normal1,axis=1)
        n = n[:, np.newaxis]
        normal1 = normal1 / n
        D1= -np.sum(normal1 * p1_, axis=1)
        p_stack = np.stack((p1_,p2_,p3_), axis=1)
        
        max_vals = np.max(p_stack, axis=1)
        min_vals = np.min(p_stack, axis=1)
        
        x_ranges = np.linspace(min_vals[:, 0], max_vals[:, 0], 10, axis=1)
        y_ranges = np.linspace(min_vals[:, 1], max_vals[:, 1], 10, axis=1)
        
        # x_range = np.linspace(min(p1[0], p2[0], p3[0]), max(p1[0], p2[0], p3[0]), 10)
        # y_range = np.linspace(min(p1[1], p2[1], p3[1]), max(p1[1], p2[1], p3[1]), 10)
        # fig = plt.figure()
        # xx, yy = np.meshgrid(x_range, y_range)
        
        xx1 = np.zeros((x_ranges.shape[0], 10, 10))
        yy1 = np.zeros((y_ranges.shape[0], 10, 10))
        
        for i in range(x_ranges.shape[0]):
            
            xx1[i], yy1[i] = np.meshgrid(x_ranges[i], y_ranges[i])
        
        A = normal1[:,0][:,np.newaxis]
        B = normal1[:,1][:,np.newaxis]
        Ax = np.einsum('ij,ijk->ijk', A, xx1)
        By = np.einsum('ij,ijk->ijk', B, yy1)
        C = normal1[:,2][:,np.newaxis][:,np.newaxis]
        D1 = D1[:,np.newaxis][:,np.newaxis]
        zz = -(Ax + By + D1)/C
        
        return zz, xx1, yy1, normal1, D1
    
    
    def project_vector_and_point_on_plane(self, p1, p2, normal, D):
        # Rzutowanie punktów i utworzonych z nich wektorów na wcześniej zdefiniowaną płaszczyznę - wersja dla wielu klatek
        v_list=[]
        v_proj_list=[]
        v_normalized_list=[]
        
        p1_proj = self.project_point_on_plane(p1 , normal, D)
        p2_proj = self.project_point_on_plane(p2 , normal, D)   
          
        v = p2 - p1
        #norms = np.linalg.norm(v, axis=1, keepdims=True)
        #v_normalized = v / norms
        #v_normalized_list.extend(v_normalized)
        v_list.extend(v)
        #dot_products = np.einsum('ij,ij->i', v_normalized, normal)
        dot_products = np.einsum('ij,ij->i', v, normal)
        v_proj = v - (dot_products[:, np.newaxis] * normal)
        v_proj_list.extend(v_proj)
        s = np.stack(v_proj_list, axis=1)
        norms1 = np.linalg.norm(v_proj, axis=1, keepdims=True)
        v_proj_normalized = v_proj / norms1

        return v_proj_normalized, v_proj, p1_proj, p2_proj
    
    
    def project_point_on_plane(self, points, normals, D):
        dot_products = np.einsum('ij,ij->i', points, normals)
        distances = (dot_products + D.flatten()) #/ np.linalg.norm(normals, axis=1)**2
        projection = points - (distances[:, np.newaxis] * normals)
        return projection
    
    
    def is_symmetric(self, sym_limbs, normal, D, xx, yy, zz):
        # Sprawdzenie jak kończyny z lewej strony są symetryczne względem prawej
        # :param sym_limbs: lista krotek o długości 2 np. (1,2)
        # :param elev: normal (Fx3) - dla wielu klatek
        # :param azim: D. (Fx1) - dla wielu klatek
        r_list = []
        p1_list = []
        p2_list = []
        p3_list = []
        p4_list = []
        v1_list = []
        v2_list = []
        for i, limb in enumerate(sym_limbs):
            v_temp=[]
            v_proj_temp=[]
            limb1 = self.limbs[limb[0]]
            limb2 = self.limbs[limb[1]]
            p1 = np.array([self.data3d[:, 0, limb1[0]], self.data3d[:, 1, limb1[0]], self.data3d[:, 2, limb1[0]]])
            p1 = np.swapaxes(p1,0,1)
            p2 = np.array([self.data3d[:, 0, limb1[1]], self.data3d[:, 1, limb1[1]], self.data3d[:, 2, limb1[1]]])
            p2 = np.swapaxes(p2,0,1)
            v_pn1, v1, p1_proj, p2_proj = self.project_vector_and_point_on_plane(p1, p2, normal, D)
            
            p3 = np.array([self.data3d[:, 0, limb2[0]], self.data3d[:, 1, limb2[0]], self.data3d[:, 2, limb2[0]]])
            p3 = np.swapaxes(p3,0,1)
            p4 = np.array([self.data3d[:, 0, limb2[1]], self.data3d[:, 1, limb2[1]], self.data3d[:, 2, limb2[1]]])
            p4 = np.swapaxes(p4,0,1)
            v_pn2, v2, p3_proj, p4_proj = self.project_vector_and_point_on_plane(p3, p4, normal, D)
            
            r = (p3_proj + v2) - (p1_proj + v1)
            r_list.append(r)
            p1_list.append(p1_proj)
            p2_list.append(p2_proj)
            p3_list.append(p3_proj)
            p4_list.append(p4_proj)
            v1_list.append(v1)
            v2_list.append(v2)
        r = np.stack(r_list, axis=2)
        p1 = np.stack(p1_list, axis=2)
        p2 = np.stack(p2_list, axis=2)
        p3 = np.stack(p3_list, axis=2)
        p4 = np.stack(p4_list, axis=2)
        v1 = np.stack(v1_list, axis=2)
        v2 = np.stack(v2_list, axis=2)
        norms = np.linalg.norm(r, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        r_normalized = r / norms
        
        return r_normalized, r, p1, p2, p3, p4, v1, v2
    
    
    def compute_trajectory(self):
        #Obliczanie trajektorii między klatkami
        displacement_vectors = np.diff(self.data3d, axis=0)
        norms = np.linalg.norm(displacement_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        unit_vectors = displacement_vectors / norms
        
        return unit_vectors, displacement_vectors 
 
    

    def avg_nodes(self, nodes):
        # oblicza środek ciężkości 1 lub więcej keypointów z self.data3d
        if isinstance(nodes, int):
            avg_node = self.data3d[:, :, nodes].reshape(self.data3d.shape[0], 3, 1)
        else:
            avg_node = np.mean(self.data3d[:, :, nodes], axis=2).reshape(self.data3d.shape[0], 3, 1)
        
        return avg_node
    
    def center_of_mass(self, gender=None):
        # nodes must be:
        # nodes = { 'Hip': 0, 'Left Hip': 1 , 'Left Knee': 2, 'Left Foot':3, 'Right Hip': 4, 'Right Knee': 5 ,
        #        'Right Foot': 6, 'Spine': 7 , 'Thorax': 8, 'Neck/Nose': 9 , 'Head': 10 , 'Right Shoulder': 11,
        #        'Right Elbow': 12, 'Right Wrist': 13 , 'Left Shoulder': 14, 'Left Elbow': 15, 'Left Wrist': 16}
        # gender = 'male' or 'female'. If not specified, takes average weights between male and female
        # weight_keypoints are expressed as single nodes or combination of 2 or more nodes, eg. (10) or (8, 7)
        # weights were calculated by de Leva in 1996, for "average" non-athlete people, both male and female
        weight_keypoints_indices = ((10), (8, 7), (7, 0), (0), (14, 15), (11, 12),
                            (15, 16), (12, 13), (16), (13), (1, 2),
                            (4, 5), (2, 3), (5, 6), (3), (6))     #   bodypart keypoints indices
        weights_m = np.array((6.94, 18.66, 12.12, 12.68, 2.71, 2.71, 1.62, 1.62,
                      0.61, 0.61, 14.16, 14.16, 4.33, 4.33, 1.37, 1.37))    #   bodypart weights for male
        weights_f = np.array((6.68, 16.03, 11.53, 15.03, 2.55, 2.55, 1.38, 1.38,
                      0.56, 0.56, 14.78, 14.78, 4.81, 4.81, 1.29, 1.29))    #   bodypart weights for female
        
        weights = {
            'male': weights_m,
            'female': weights_f,
            None: (weights_m + weights_f) / 2
        }[gender]

        weights_norm = weights / np.sum(weights)

        new_keypoints_list = []
        for indices in weight_keypoints_indices:
            weight_keypoint = self.avg_nodes(indices)
            new_keypoints_list.append(weight_keypoint)

        weight_keypoints = np.concatenate(new_keypoints_list, axis=2)

        weighted_keypoints = weights_norm * weight_keypoints
        center_of_mass = np.sum(weighted_keypoints, axis=2)
        
        return center_of_mass

    def gravity_angle(self, nodes, contact_point, gender=None):
        # nodes = keypointy, których kąt od wektora grawitacji liczyć
        # contact_point = node lub krotka nodów, które są punktem podparcia (np. (3,6) dla midpoint stóp)
        # gender = 'male' lub 'female'
        cent_mass = self.center_of_mass(gender)
        cont_points = self.avg_nodes(contact_point).squeeze(2)
        
        angles_list = []
        degree_list = []
        for node in nodes:
            node_point = self.data3d[:, :, node]
            joints_list = [np.stack((node_point, cent_mass, cont_points), axis=-1)]
            thetas_degrees, angle_cos = self.calc_angle(joints_list, mode='absolute')
            angles_list.append(angle_cos)
            degree_list.append(thetas_degrees)
        
        angles = np.concatenate(angles_list, axis=1)
        angles_degree = np.concatenate(degree_list, axis=1)
        
        return angles, angles_degree, cent_mass, cont_points
    