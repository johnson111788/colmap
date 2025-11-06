import tqdm
import os
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import time
import random
import fcntl
import h5py

from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d


def safe_hloc_operation(func, *args, max_retries=5, **kwargs):
    """安全执行hloc操作，带重试机制"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (BlockingIOError, OSError, PermissionError) as e:
            if any(msg in str(e).lower() for msg in ["resource temporarily unavailable", "unable to lock file", "errno = 11"]) and attempt < max_retries - 1:
                # 随机等待以避免同时重试
                wait_time = random.uniform(2, 8)
                print(f"文件锁冲突，等待 {wait_time:.1f}s 后重试 (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                raise e
        except Exception as e:
            # 其他异常直接抛出
            raise e
    

def process_uuid_folder_on_gpu(args):
    """在指定GPU上处理单个UUID文件夹的函数"""
    uuid_folder_path, uuid_name, gpu_id, skip_geometric_verification, outputs_base_dir = args
    
    # 设置当前进程使用的GPU设备
    process_id = os.getpid()
    if torch.cuda.is_available():
        gpu_device = gpu_id % torch.cuda.device_count()
        torch.cuda.set_device(gpu_device)
        # 确保当前设备被正确设置
        current_device = torch.cuda.current_device()
        print(f"[GPU {current_device}|PID {process_id}] 已设置CUDA设备到GPU {gpu_device}")
    else:
        gpu_device = 0
        print(f"[CPU|PID {process_id}] CUDA不可用，使用CPU")
    
    print(f"[GPU {gpu_device}|PID {process_id}] 处理文件夹: {uuid_name}")
    
    # 设置路径
    images = Path(uuid_folder_path)
    outputs = Path(outputs_base_dir) / uuid_name
    
    # 创建输出目录
    outputs.mkdir(parents=True, exist_ok=True)
    
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    
    # 确保输出目录存在并且是独立的
    sfm_dir.mkdir(parents=True, exist_ok=True)

    feature_conf = extract_features.confs["aliked-n16"]
    matcher_conf = match_features.confs["aliked+lightglue"]

    # 获取该文件夹下的所有图片文件
    image_files = [f for f in images.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    if not image_files:
        print(f"[GPU {gpu_device}|PID {process_id}] 警告: {uuid_name} 文件夹中没有找到图片文件")
        return False, uuid_name, f"没有找到图片文件"
    
    # 生成相对路径的图片列表
    references = [p.relative_to(images).as_posix() for p in image_files]
    print(f"[GPU {gpu_device}|PID {process_id}] 找到 {len(references)} 张图片")

    try:
        start_time = time.time()
        
        # 特征提取
        print(f"[GPU {gpu_device}|PID {process_id}] 开始特征提取: {uuid_name}")
        safe_hloc_operation(
            extract_features.main,
            feature_conf, images, image_list=references, feature_path=features
        )
        
        # 生成图片对
        print(f"[GPU {gpu_device}|PID {process_id}] 生成图片对: {uuid_name}")
        safe_hloc_operation(
            pairs_from_exhaustive.main, sfm_pairs, image_list=references
        )
        
        # 特征匹配
        print(f"[GPU {gpu_device}|PID {process_id}] 开始特征匹配: {uuid_name}")
        safe_hloc_operation(
            match_features.main, matcher_conf, sfm_pairs, features=features, matches=matches
        )

        # 3D重建
        if skip_geometric_verification:
            print(f"[GPU {gpu_device}|PID {process_id}] 开始3D重建 (跳过几何验证): {uuid_name}")
        else:
            print(f"[GPU {gpu_device}|PID {process_id}] 开始3D重建 (包含几何验证): {uuid_name}")
            
        model = safe_hloc_operation(
            reconstruction.main,
            sfm_dir, images, sfm_pairs, features, matches,
            image_list=references,
            skip_geometric_verification=skip_geometric_verification,
            # min_reg_images=60
        )
        
        # 生成3D可视化
        print(f"[GPU {gpu_device}|PID {process_id}] 生成3D可视化: {uuid_name}")
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
        )
        
        # 保存HTML文件
        html_path = f"html/{uuid_name}.html"
        fig.write_html(html_path)
        
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_device}|PID {process_id}] 完成处理 {uuid_name}，耗时: {elapsed:.2f}秒")
        
        return True, uuid_name, f"成功处理，耗时: {elapsed:.2f}秒"
        
    except Exception as e:
        error_msg = f"处理 {uuid_name} 时出错: {str(e)}"
        print(f"[GPU {gpu_device}|PID {process_id}] {error_msg}")
        return False, uuid_name, error_msg


def process_multiple_folders_on_gpu(folder_batch, gpu_id, skip_geometric_verification=True, outputs_base_dir="outputs.train"):
    """在单个GPU上处理多个文件夹的函数"""
    results = []
    for uuid_folder_path, uuid_name in folder_batch:
        args = (uuid_folder_path, uuid_name, gpu_id, skip_geometric_verification, outputs_base_dir)
        result = process_uuid_folder_on_gpu(args)
        results.append(result)
        
        # 释放GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def distribute_folders_to_gpus(uuid_folders, num_gpus=8, processes_per_gpu=4):
    """将文件夹分配到多个GPU和进程"""
    total_processes = num_gpus * processes_per_gpu
    folder_chunks = [[] for _ in range(total_processes)]
    
    # 将文件夹均匀分配到所有进程
    for i, folder in enumerate(uuid_folders):
        chunk_idx = i % total_processes
        folder_chunks[chunk_idx].append((folder, folder.name))
    
    # 创建GPU分配信息
    gpu_assignments = []
    for gpu_id in range(num_gpus):
        for proc_id in range(processes_per_gpu):
            chunk_idx = gpu_id * processes_per_gpu + proc_id
            if folder_chunks[chunk_idx]:  # 只处理非空的chunk
                gpu_assignments.append((folder_chunks[chunk_idx], gpu_id))
    
    return gpu_assignments

# python demo.py >> demo.log 2>&1
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HLOC视频重建 - 支持批量和增量处理"
    )
    parser.add_argument(
        "--data-base-path",
        type=Path,
        default=Path("YOUR_DATA_PATH-frames_4fps"),
        help="原始数据目录"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("YOUR_DATA_PATH-hloc"),
        help="HLOC输出目录"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="GPU数量"
    )
    parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=4,
        help="每个GPU的进程数"
    )
    parser.add_argument(
        "--skip-geometric-verification",
        action="store_true",
        help="跳过几何验证以提升速度"
    )
    parser.add_argument(
        "--target-cases",
        type=str,
        default=None,
        help="逗号分隔的目标case ID列表（如果指定，只处理这些case）"
    )
    
    args = parser.parse_args()
    
    # 配置参数
    NUM_GPUS = args.num_gpus
    PROCESSES_PER_GPU = args.processes_per_gpu
    SKIP_GEOMETRIC_VERIFICATION = args.skip_geometric_verification
    
    # 指定数据目录
    data_base_path = args.data_base_path
    
    # 获取所有UUID文件夹
    if args.target_cases:
        # 只处理指定的cases
        target_case_ids = set(args.target_cases.split(','))
        uuid_folders = [data_base_path / case_id for case_id in target_case_ids if (data_base_path / case_id).is_dir()]
        print(f"目标处理 {len(target_case_ids)} 个指定的case，找到 {len(uuid_folders)} 个有效文件夹")
    else:
        # 处理所有文件夹
        uuid_folders = [f for f in data_base_path.iterdir() if f.is_dir()]
        print(f"找到 {len(uuid_folders)} 个UUID文件夹")
    
    print(f"找到 {len(uuid_folders)} 个UUID文件夹")
    print(f"使用 {NUM_GPUS} 个GPU，每个GPU运行 {PROCESSES_PER_GPU} 个进程")
    print(f"总并行度: {NUM_GPUS * PROCESSES_PER_GPU}")
    print(f"几何验证设置: {'跳过' if SKIP_GEOMETRIC_VERIFICATION else '启用'}")
    if SKIP_GEOMETRIC_VERIFICATION:
        print("⚠️  跳过几何验证可大幅提升速度，但可能影响3D重建质量")
    
    # 分配任务到GPU
    gpu_assignments = distribute_folders_to_gpus(uuid_folders, NUM_GPUS, PROCESSES_PER_GPU)
    print(f"创建了 {len(gpu_assignments)} 个工作任务")
    
    # 使用进程池并行处理
    total_start_time = time.time()
    all_results = []
    
    with ProcessPoolExecutor(max_workers=len(gpu_assignments)) as executor:
        # 提交所有任务
        future_to_gpu = {
            executor.submit(process_multiple_folders_on_gpu, folder_batch, gpu_id, SKIP_GEOMETRIC_VERIFICATION, str(args.outputs_dir)): gpu_id 
            for folder_batch, gpu_id in gpu_assignments
        }
        
        # 收集结果
        with tqdm.tqdm(total=len(uuid_folders), desc="处理进度") as pbar:
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    pbar.update(len(results))
                except Exception as exc:
                    print(f'GPU {gpu_id} 产生异常: {exc}')
    
    # 统计结果
    total_elapsed = time.time() - total_start_time
    successful = sum(1 for success, _, _ in all_results if success)
    failed = len(all_results) - successful
    
    print(f"\n=== 处理完成 ===")
    print(f"总耗时: {total_elapsed:.2f}秒")
    print(f"成功处理: {successful} 个文件夹")
    print(f"处理失败: {failed} 个文件夹")
    print(f"平均每个文件夹耗时: {total_elapsed/len(uuid_folders):.2f}秒")
    
    # 显示失败的文件夹
    if failed > 0:
        print(f"\n失败的文件夹:")
        for success, uuid_name, message in all_results:
            if not success:
                print(f"  - {uuid_name}: {message}")
    
    print(f"\nHTML文件保存在 ./html/ 目录下")