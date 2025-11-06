#!/usr/bin/env python3
"""
将HLOC数据转换为VGGT格式

基于分析结果：
- VGGT格式: 
  * extrinsic: (N, 3, 4) - N帧的3x4投影矩阵 [R|t] (w2c格式)
  * intrinsic: (N, 3, 3) - N帧的3x3内参矩阵
- HLOC格式: 单独的相机和图像数据

此脚本将HLOC的相机参数转换为VGGT的序列格式
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加hloc模块到Python路径
current_dir = Path(__file__).parent
hloc_path = current_dir / "hloc"
if hloc_path.exists():
    sys.path.insert(0, str(current_dir))

from hloc.utils.read_write_model import read_cameras_binary, read_images_binary


def qvec2rotmat(qvec):
    """四元数转旋转矩阵"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def convert_hloc_to_vggt_sequence(sfm_dir="outputs/0df0f621-205e-4b48-8832-fdccddc5509c/sfm", output_prefix="hloc_sequence_corrected"):
    """将HLOC数据转换为VGGT序列格式
    
    修正说明:
    1. 位置修正: 平移向量t取反 
    2. 朝向修正: 旋转矩阵xyz轴取反（修正orientation方向）
    """
    
    cameras_path = os.path.join(sfm_dir, "cameras.bin")
    images_path = os.path.join(sfm_dir, "images.bin")
    
    # 加载HLOC数据
    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    
    # 按图像ID排序，确保序列一致性
    sorted_images = sorted(images.items(), key=lambda x: x[0])
    
    num_frames = len(sorted_images)
    # import ipdb; ipdb.set_trace()
    # 初始化序列数组
    extrinsics_sequence = np.zeros((num_frames, 3, 4), dtype=np.float32)
    intrinsics_sequence = np.zeros((num_frames, 3, 3), dtype=np.float32)
    
    for idx, (image_id, image) in enumerate(sorted_images):
        camera = cameras[image.camera_id]
        
        # 转换外参: qvec, tvec -> [R|t] (3x4)
        R = qvec2rotmat(image.qvec)  # 3x3
        t = image.tvec.reshape(3, 1)  # 3x1
        
        # 坐标系转换: HLOC -> VGGT 
        # 根据可视化验证结果：
        
        # 构建3x4投影矩阵 [R|t]
        extrinsic_3x4 = np.hstack([R, t])  # (3, 4)
        extrinsics_sequence[idx] = extrinsic_3x4
        
        # 转换内参: SIMPLE_RADIAL -> K矩阵
        if camera.model == "SIMPLE_RADIAL":
            f = camera.params[0]   # 焦距
            cx = camera.params[1]  # 主点x
            cy = camera.params[2]  # 主点y
            # k1 = camera.params[3]  # 畸变系数(在K矩阵中不包含)
            
            K = np.array([
                [f,  0,  cx],
                [0,  f,  cy],
                [0,  0,  1 ]
            ], dtype=np.float32)
            
            intrinsics_sequence[idx] = K
        else:
            print(f"警告: 相机{camera.camera_id}使用不支持的模型{camera.model}")
    
    print("extrinsic_sequence shape: ", extrinsics_sequence.shape)
    # 保存为VGGT格式
    extrinsic_file = f"{output_prefix}_extrinsic.npy"
    intrinsic_file = f"{output_prefix}_intrinsic.npy"
    
    np.save(extrinsic_file, extrinsics_sequence)
    np.save(intrinsic_file, intrinsics_sequence)
        
    return extrinsics_sequence, intrinsics_sequence





def process_all_outputs(outputs_dir="outputs", camera_output_dir="camera", target_cases=None):
    """
    遍历所有outputs文件夹，为每个UUID生成相机参数文件
    
    Args:
        outputs_dir: 包含所有UUID文件夹的输出目录
        camera_output_dir: 相机参数文件的输出目录
        target_cases: 逗号分隔的目标case ID列表字符串（可选）
    """
    print("="*80)
    print("HLOC到VGGT格式批量转换工具")
    print("="*80)
    
    # 创建camera输出目录
    os.makedirs(camera_output_dir, exist_ok=True)
    print(f"📁 输出目录: {camera_output_dir}/")
    
    # 获取所有UUID文件夹
    if not os.path.exists(outputs_dir):
        print(f"❌ 错误: 输出目录 {outputs_dir} 不存在!")
        return
    
    # 如果指定了target_cases，只处理这些cases
    if target_cases:
        target_case_ids = set(target_cases.split(','))
        uuid_folders = []
        for case_id in target_case_ids:
            uuid_path = os.path.join(outputs_dir, case_id)
            sfm_path = os.path.join(uuid_path, "sfm")
            if os.path.isdir(uuid_path) and os.path.exists(sfm_path):
                uuid_folders.append(case_id)
        print(f"🔍 目标处理 {len(target_case_ids)} 个指定的case，找到 {len(uuid_folders)} 个有效的UUID文件夹")
    else:
        # 处理所有文件夹
        uuid_folders = []
        for item in os.listdir(outputs_dir):
            uuid_path = os.path.join(outputs_dir, item)
            sfm_path = os.path.join(uuid_path, "sfm")
            if os.path.isdir(uuid_path) and os.path.exists(sfm_path) and os.path.exists(os.path.join('camera.train_fixed.tighter.v1', item+"_extrinsic.npy")): #  and os.path.exists(os.path.join('camera.train_fixed', item+"_extrinsic.npy"))
                uuid_folders.append(item)
        print(f"🔍 发现 {len(uuid_folders)} 个有效的UUID文件夹")
    
    if not uuid_folders:
        print(f"❌ 在 {outputs_dir} 中未找到有效的UUID文件夹!")
        return
    
    # 统计信息
    success_count = 0
    failed_count = 0
    # uuid_folders = ['0df0f621-205e-4b48-8832-fdccddc5509c']
    
    # 处理每个UUID文件夹
    for i, uuid in enumerate(uuid_folders, 1):
        print(f"\n" + "="*60)
        print(f"处理 [{i}/{len(uuid_folders)}]: {uuid}")
        print("="*60)
        
        try:
            # 构建路径
            sfm_dir = os.path.join(outputs_dir, uuid, "sfm")
            
            # 转换序列
            hloc_ext, hloc_int = convert_hloc_to_vggt_sequence(
                sfm_dir=sfm_dir, 
                output_prefix=f"{camera_output_dir}/{uuid}"
            )
            
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            print(f"❌ {uuid} 处理失败: {str(e)}")
            continue
    
    # 最终统计
    print(f"\n" + "="*80)
    print("批量处理完成")
    print("="*80)
    print(f"📊 处理统计:")
    print(f"   ✅ 成功: {success_count}")
    print(f"   ❌ 失败: {failed_count}")
    print(f"   📁 总计: {len(uuid_folders)}")
    print(f"💾 所有文件已保存到: {camera_output_dir}/")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HLOC到VGGT格式转换 - 支持批量和增量处理"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="YOUR_DATA_PATH-hloc",
        help="HLOC输出目录"
    )
    parser.add_argument(
        "--camera-dir",
        type=str,
        default="YOUR_DATA_PATH-camera",
        help="相机参数输出目录"
    )
    parser.add_argument(
        "--target-cases",
        type=str,
        default=None,
        help="逗号分隔的目标case ID列表（如果指定，只处理这些case）"
    )
    
    args = parser.parse_args()
    
    # 处理所有outputs文件夹或指定的cases
    process_all_outputs(args.outputs_dir, args.camera_dir, args.target_cases)

if __name__ == "__main__":
    main()
