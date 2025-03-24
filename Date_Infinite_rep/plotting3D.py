import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers  
import os
from functools import partial
import plotting3D
    
def plot_3d_pose(ax, pose, limbs, frame, limbs_angle = None,  angles=None, xx=None, yy=None, zz=None):
    pose = pose.flatten(order='F')
    vals = np.reshape(pose, (17, -1))
    artists = []
    
    left_limb = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2])
    for i, limb in enumerate(limbs):
        x, y, z = [np.array([vals[limb[0], j], vals[limb[1], j]]) for j in range(3)]
        if left_limb[i] == 0:
            cc = 'blue'
        elif left_limb[i] == 1:
            cc = 'red'
        else:
            cc = 'black'
        lines = ax.plot(x, y, z, marker='o', markersize=2, lw=   1, c=cc)
        text1 = ax.text(x[0], y[0], z[0], s = f'{limb[0]}', ha='center')
        text2 = ax.text(x[1], y[1], z[1], s = f'{limb[1]}', ha='center')
        artists.extend(lines)
        artists.extend([text1])
        artists.extend([text2])
    
    if limbs_angle is not None and angles is not None: 
        # Plot angles between specified limbs
        for i, limb in enumerate(limbs_angle):
            A, B, C = limb
            angle = angles[frame,[i]]

            # Rysowanie półokręgu przedstawiającego kąt
            arc, angle1 = plot_arc(ax, vals[A], vals[B], vals[C])
        
            # Rysowanie tekstu przedstawiającego wartość kąta
            text_angle = ax.text(vals[B][0]+30, vals[B][1]+30, vals[B][2]+30, s=f'{angle[0]:.1f}°', ha='center', color='green')
            artists.extend([text_angle])
            artists.extend(arc)

    return artists


def plot_g_angle(ax, pose, g_nodes, angles_degree, frame, cent_mass, cont_points):
    artists = []
    for i, node in enumerate(g_nodes[0:2]):
        g_node = pose[frame,:,node]
        angle = angles_degree[frame, i]
        center = cent_mass[frame]
        cont_point = cont_points[frame]
        # Rysowanie półokręgu przedstawiającego kąt
        arc, angle1 = plot_arc(ax, g_node, center, cont_point)
        scat_points1 = ax.scatter(g_node[0], g_node[1], g_node[2], color='y', s=15)
        scat_points2 = ax.scatter(center[0], center[1], center[2], color='b', s=20)
        scat_points3 = ax.scatter(cont_point[0], cont_point[1], cont_point[2], color='r', s=20)

        # Rysowanie tekstu przedstawiającego wartość kąta
        text_angle = ax.text(center[0]+15, center[1]+15, center[2]+15, s=f'{angle:.1f}°', ha='center', color='green')
        artists.extend([text_angle])
        artists.extend(arc)
        artists.extend([scat_points1])
        artists.extend([scat_points2])
        artists.extend([scat_points3])

    return artists


def visualize_sym_vectors(r, p1_proj, p2_proj, p3_proj, p4_proj, v1_proj, v2_proj, frame, D, normal, xx=None, yy=None, zz=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if xx is None:
        d1 = np.linspace(100, 200, 100)
        d = np.linspace(500, 550, 50)
        x, y = np.meshgrid(d1, d)
        A = normal[frame,0]
        B = normal[frame,1]
        #z1 = -(normal1[:,0]*xx + normal1[1]*yy +D1) / normal[2] 
        Ax = A*x
        By = B*y
        C = normal[0,2]
        D = D[frame,0,0]
        zz = -(Ax + By + D)/C
    else:
        x = xx[frame,:,:]
        y = yy[frame,:,:]
        zz = zz[frame,:,:]
        
    ax.plot_surface(x, y, zz, alpha=0.5)
    
    for i in range(p1_proj.shape[2]):
        start1 = p1_proj[frame,:, i]
        end1 = start1 + v1_proj[frame,:, i]
        proj1 = np.stack((p1_proj[frame,:, i], p2_proj[frame,:, i]))
        
        start2 = p3_proj[frame, :, i]
        end2 = start2 + v2_proj[frame, :, i]
        proj2 = np.stack((p3_proj[frame,:, i],p4_proj[frame, :, i]))
        end3 = end1 + r[frame, :, i]
    
        ax.plot(proj1[:,0], proj1[:,1], proj1[:,2], marker='o', markersize=2, lw=   1, c='b')
        ax.scatter(proj1[:,0], proj1[:,1], proj1[:,2], color='b', s=10)
        ax.quiver(start1[0], start1[1], start1[2], end1[0] - start1[0], end1[1] - start1[1], end1[2] - start1[2], color='m')
        ax.plot(proj2[:,0], proj2[:,1], proj2[:,2], marker='o', markersize=2, lw=   1, c='y')
        ax.scatter(proj2[:,0], proj2[:,1], proj2[:,2], color='y', s=10)
        ax.quiver(start2[0], start2[1], start2[2], end2[0] - start2[0], end2[1] - start2[1], end2[2] - start2[2], color='c')
        ax.quiver(end1[0], end1[1], end1[2], end3[0] - end1[0], end3[1] - end1[1], end3[2] - end1[2], color='g', linestyle='dashed')


    # Ustawienia osi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax 
 

def plot_arc(ax, A, B, C, radius=25, num_points=100):
    """Rysuje półokrąg między punktami A i C z punktem B jako centrum łuku"""
    BA = A - B
    BC = C - B

    # Normalizacja wektorów
    BA = BA / np.linalg.norm(BA)
    BC = BC / np.linalg.norm(BC)

    # Znalezienie wektora normalnego do płaszczyzny wyznaczonej przez BA i BC
    normal = np.cross(BA, BC)
    normal = normal / np.linalg.norm(normal)

    # Znalezienie wektora w płaszczyźnie, który jest prostopadły do BA
    v = np.cross(normal, BA)
    v = v / np.linalg.norm(v)

    # Kąt między BA i BC
    angle = np.arccos(np.dot(BA, BC))
    angle_deg = np.degrees(angle)
    theta = np.linspace(0, angle, num_points)

    # Generowanie punktów na łuku
    arc_points = [B + radius * (np.cos(t) * BA + np.sin(t) * v) for t in theta]
    arc_points = np.array(arc_points)
    
    # Rysowanie łuku
    arc = ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 'g-')

    return arc, angle_deg

def plot_trajectory(ax, pose, displacement, frame):
    _displacement = displacement[frame, :, :]
    for i in range(_displacement.shape[1]):
        start = pose[frame, :, i]
        end = start + _displacement[:, i]*1.5
        ax.quiver(start[0], start[1], start[2], end[0] - start[0], end[1] - start[1], end[2] - start[2], color='y')
        
    return ax
    
def animation_frame(data_3d, limbs, limbs_angle=None, angles=None, xx=None, yy=None, zz=None):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,projection='3d')
    plot_3d_pose(ax, pose=data_3d, frame=0, limbs=limbs, limbs_angle=limbs_angle, angles=angles,xx=xx, yy=yy, zz=zz)
    plt.show()
    
    
def animate(frame, axes, data3d,  limbs, limbs_angle = None, angles=None, xx=None, yy=None, zz=None, data3d_t=None, angles_t=None):
    
    artists = []
    if len(axes) == 2:
        ax1, ax2 = axes
    else:
        ax = axes[0]

    if data3d is not None:
        current_ax = ax1 if 'ax1' in locals() else ax
        current_ax.clear()
        current_ax.set_title(f"Bad habits, frame: {frame}")
        current_ax.set_xlabel('X')
        current_ax.set_ylabel('Y')
        current_ax.set_zlabel('Z')
        current_ax.set_xlim((-1000, 1000))
        current_ax.set_ylim((-1000, 1000))
        current_ax.set_zlim((0, 2000))

        data = data3d[frame, :, :]
        artists1 = plot_3d_pose(current_ax, pose=data, frame=frame, limbs=limbs,
                                limbs_angle=limbs_angle, angles=angles, xx=xx, yy=yy, zz=zz)
        artists.extend(artists1)

    if data3d_t is not None:
        current_ax = ax2 if 'ax2' in locals() else ax
        current_ax.clear()
        current_ax.set_title(f"True, frame: {frame}")
        current_ax.set_xlabel('X')
        current_ax.set_ylabel('Y')
        current_ax.set_zlabel('Z')
        current_ax.set_xlim((-1000, 1000))
        current_ax.set_ylim((-1000, 1000))
        current_ax.set_zlim((0, 2000))

        data_t = data3d_t[frame, :, :]
        artists2 = plot_3d_pose(current_ax, pose=data_t, frame=frame, limbs=limbs,
                                limbs_angle=limbs_angle, angles=angles_t, xx=xx, yy=yy, zz=zz)
        artists.extend(artists2)
        
    f_max = data3d.shape[0]
    # if frame % 30 == 0 or frame == f_max:
    #     print(f'Grabbing frame no. {frame}/{f_max}')
    
        
    plt.show()
    return artists
    
    
def animation_frames(name, data3d, limbs, limbs_angle = None, angles=None, xx=None, yy=None, zz=None, data3d_t=None, angles_t=None):
    
    if data3d is None and data3d_t is None:
        raise ValueError("Both data3d and data3d_t cannot be None.")
    
    if data3d is not None and data3d_t is not None:
        fig = plt.figure(figsize=(16, 8))  # Dwa wykresy obok siebie
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        axes = [ax1, ax2]
        f = min(data3d.shape[0], data3d_t.shape[0])  # Użyj minimalnej liczby klatek
    else:
        fig = plt.figure(figsize=(8, 8))  # Jeden wykres
        ax = fig.add_subplot(111, projection='3d')
        axes = [ax]
        f = data3d.shape[0] if data3d is not None else data3d_t.shape[0]
        
    f = data3d.shape[0]
    for frame in range(40):
        animate(50, axes, data3d,  limbs, limbs_angle, angles, xx, yy, zz, data3d_t, angles_t)
        plt.ion()
        plt.draw()
        plt.show()
        # plt.pause(1)
    
    # anim= FuncAnimation(fig, animate, frames=f, fargs=(axes, data3d,  limbs, limbs_angle, angles, xx, yy, zz, data3d_t, angles_t),  interval=100/3, blit=True)
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=10, metadata=dict(artist='Adam'), bitrate=1800)
    # anim.save(name, writer=writer
    print()
    
def loss_heatmap(data_arr, plot_title="", plot_show=True, plot_block=True, plot_save=False, plot_dir="", vmin=0, vmax=0.5, width_px=2560, height_px=1440, dpi=100):
    #   data_arr - array (MxN) gdzie M to zazwyczaj liczba klatek, a N liczba cech
    #   plot_show - czy rysować plot i czekać na zamknięcie przez użytkownika
    #   plot_block - czy czekać z wykonywaniem kodu dalej aż użytkownik zamknie wykres
    #   plot_save - czy zapisać plot
    #   plot_dir - ścieżka do folderu w którym zapisać plot
    #   vmax, vmin - zakres wartości dla skali heatmapy
    #   width_px, height_px, dpi - parametry do wyświetlania i zapisu plotu 
    
    fig_width = width_px / dpi
    fig_height = height_px / dpi
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(data_arr.T, aspect='auto', cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Loss')
    plt.xlabel('Czas (klatki)')
    plt.ylabel('Cechy outputu modelu')
    plt.yticks(range(data_arr.shape[1]), [f'KP{i+1}' for i in range(data_arr.shape[1])])
    plt.title(plot_title)
    plt.tight_layout()
        
    if plot_save == True:    
        plt.savefig(os.path.join(plot_dir, plot_title, '.png'), dpi=dpi)
        
    if plot_show == True:
        plt.show(block=plot_block)
    else:
        plt.close()
          
    
    
   
    

















