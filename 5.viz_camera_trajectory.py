import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import sys
from tqdm import tqdm

data_path = 'YOUR_DATA_PATH-camera'
file_list = [file for file in os.listdir(data_path) if file.endswith('_extrinsic.npy')] # len(file.split('_')) == 2 and 

os.makedirs(f'{data_path}/camera_trajectory', exist_ok=True)
for file in tqdm(file_list):
    save_path = f'{data_path}/camera_trajectory/{file.replace("/", "_").replace(".npy", ".png")}'
    if os.path.exists(save_path):
        continue
    extrinsic_clip = np.load(os.path.join(data_path, file))#.squeeze(1)
    extrinsic_clip_path = file

    # 去掉多余的维度以方便处理
    extrinsic_clip_squeezed = extrinsic_clip#.squeeze(1)  # [20, 3, 4]
    print(f"extrinsic_clip_squeezed shape: {extrinsic_clip_squeezed.shape}")

    # 原始數據的相機中心位置（w2c格式，需要转换为相机中心位置）
    # 从w2c转换到相机中心位置: camera_center = -R^T * t
    R_w2c = extrinsic_clip_squeezed[:, :3, :3]  # [20, 3, 3]
    t_w2c = extrinsic_clip_squeezed[:, :3, 3]   # [20, 3]
    original_cam_centers_opencv = -(R_w2c.transpose(0, 2, 1) @ t_w2c[..., np.newaxis]).squeeze(-1)  # [20, 3]
    
    # 坐標系轉換：從OpenCV (x-right, y-down, z-forward) 到直觀坐標系 (x-right, y-forward, z-up)
    # 轉換矩陣: [1, 0, 0; 0, 0, 1; 0, -1, 0]
    original_cam_centers = original_cam_centers_opencv.copy()
    original_cam_centers[:, 1] = original_cam_centers_opencv[:, 2]   # z_opencv -> y_intuitive  
    original_cam_centers[:, 2] = -original_cam_centers_opencv[:, 1]  # -y_opencv -> z_intuitive

    # 畫出相機軌跡
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 畫原始數據點（不透明）
    print(f"original_cam_centers shape: {original_cam_centers.shape}")
    ax.scatter(original_cam_centers[:, 0], original_cam_centers[:, 1], original_cam_centers[:, 2], 
            c='blue', s=40, alpha=1.0, label='Original Points', edgecolors='black', linewidth=0.5)

    # 添加相機方向箭頭（僅針對原始數據點）
    for i in range(len(original_cam_centers)):
        R_w2c = extrinsic_clip_squeezed[i, :3, :3]  # w2c 旋轉矩陣
        t_w2c = extrinsic_clip_squeezed[i, :3, 3]   # w2c 平移向量
        # 從w2c轉換為c2w來計算相機方向
        R_c2w_opencv = R_w2c.T  # c2w 旋轉矩陣 (OpenCV坐標系)
        cam_center = original_cam_centers[i]  # 相機中心位置 (已轉換到直觀坐標系)
        
        # 在OpenCV坐標系中，相機前向是+Z方向
        forward_opencv = R_c2w_opencv @ np.array([0, 0, 1])
        
        # 將方向向量也轉換到直觀坐標系
        # forward = np.array([-forward_opencv[0], -forward_opencv[2], forward_opencv[1]])*5
        forward = np.array([forward_opencv[0], forward_opencv[2], -forward_opencv[1]])*5
        
        
        ax.quiver(cam_center[0], cam_center[1], cam_center[2],
                forward[0], forward[1], forward[2],
                length=0.2, color='red', normalize=False, alpha=0.8, arrow_length_ratio=0.3)

    # 標記起點和終點
    ax.scatter(original_cam_centers[0, 0], original_cam_centers[0, 1], original_cam_centers[0, 2], 
            c='green', s=100, alpha=1.0, label='Start', edgecolors='black', linewidth=1)
    ax.scatter(original_cam_centers[-1, 0], original_cam_centers[-1, 1], original_cam_centers[-1, 2], 
            c='red', s=100, alpha=1.0, label='End', edgecolors='black', linewidth=1)

    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Forward)')  
    ax.set_zlabel('Z (Up)')
    ax.set_title(f'Camera Trajectory with Interpolation and Orientation: {extrinsic_clip_path}')
    ax.legend()
    
    # 设置相同的坐标轴比例（以最大轴范围为基准）
    x_range = original_cam_centers[:, 0].max() - original_cam_centers[:, 0].min()
    y_range = original_cam_centers[:, 1].max() - original_cam_centers[:, 1].min()
    z_range = original_cam_centers[:, 2].max() - original_cam_centers[:, 2].min()
    max_range = max(x_range, y_range, z_range) / 2.0
    
    x_mid = (original_cam_centers[:, 0].max() + original_cam_centers[:, 0].min()) * 0.5
    y_mid = (original_cam_centers[:, 1].max() + original_cam_centers[:, 1].min()) * 0.5
    z_mid = (original_cam_centers[:, 2].max() + original_cam_centers[:, 2].min()) * 0.5
    
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()