#!/usr/bin/env python3
"""
修复模型选择问题的脚本

这个脚本会：
1. 扫描所有 outputs/ 目录中的重建结果
2. 检查每个输出中的所有模型（sfm/ 和 models/ 子目录）
3. 找到注册图像数量最多的模型
4. 将正确的模型文件移动到 sfm/ 目录
"""

import pycolmap
import shutil
from pathlib import Path
import sys
import argparse
from typing import Dict, List, Tuple, Optional


def read_reconstruction_safe(model_path: Path) -> Optional[pycolmap.Reconstruction]:
    """安全地读取重建模型"""
    try:
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(str(model_path))
        return reconstruction
    except Exception as e:
        print(f"    读取 {model_path} 失败: {e}")
        return None


def analyze_output_folder(output_dir: Path) -> Tuple[Optional[str], int, Dict[str, int]]:
    """
    分析一个输出文件夹，找到最大的模型
    
    Returns:
        (best_model_path, max_images, all_models_info)
    """
    print(f"\n分析输出目录: {output_dir.name}")
    
    models_info = {}
    max_images = 0
    best_model = None
    
    sfm_dir = output_dir / "sfm"
    if not sfm_dir.exists():
        print(f"  跳过 {output_dir.name}: 没有 sfm 目录")
        return None, 0, {}
    
    # 1. 检查 sfm/ 目录本身
    print("  检查 sfm/ 目录:")
    sfm_reconstruction = read_reconstruction_safe(sfm_dir)
    if sfm_reconstruction:
        num_images = sfm_reconstruction.num_reg_images()
        models_info["sfm"] = num_images
        print(f"    sfm/: {num_images} 个注册图像")
        if num_images > max_images:
            max_images = num_images
            best_model = "sfm"
    
    # 2. 检查 models/ 子目录
    models_path = sfm_dir / "models"
    if models_path.exists():
        print("  检查 models/ 子目录:")
        for model_dir in models_path.iterdir():
            if model_dir.is_dir() and model_dir.name.isdigit():
                model_name = f"models/{model_dir.name}"
                
                # 先检查是否有重建文件，避免读取空文件夹
                filenames = ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]
                has_files = any((model_dir / filename).exists() for filename in filenames)
                
                if has_files:
                    reconstruction = read_reconstruction_safe(model_dir)
                    if reconstruction:
                        num_images = reconstruction.num_reg_images()
                        models_info[model_name] = num_images
                        print(f"    {model_name}: {num_images} 个注册图像")
                        if num_images > max_images:
                            max_images = num_images
                            best_model = model_name
                else:
                    print(f"    {model_name}: 空文件夹（跳过）")
    
    print(f"  最佳模型: {best_model} ({max_images} 个图像)")
    return best_model, max_images, models_info


def find_empty_model_folder(models_dir: Path) -> Optional[Path]:
    """找到空的 models 文件夹"""
    if not models_dir.exists():
        return None
    
    filenames = ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # 检查这个文件夹是否为空（没有重建文件）
            has_files = any((model_dir / filename).exists() for filename in filenames)
            if not has_files:
                return model_dir
    
    return None


def backup_sfm_to_models(sfm_dir: Path) -> Optional[Path]:
    """将当前的 sfm 文件备份到空的 models 文件夹中"""
    models_dir = sfm_dir / "models"
    empty_folder = find_empty_model_folder(models_dir)
    
    if not empty_folder:
        print("  警告: 没有找到空的 models 文件夹，无法备份原始 sfm 文件")
        return None
    
    filenames = ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]
    
    for filename in filenames:
        source_file = sfm_dir / filename
        backup_file = empty_folder / filename
        if source_file.exists():
            shutil.copy2(str(source_file), str(backup_file))
    
    print(f"  原始 sfm 文件已备份到: models/{empty_folder.name}")
    return empty_folder


def move_model_to_sfm(output_dir: Path, best_model: str, dry_run: bool = False) -> bool:
    """将最佳模型移动到 sfm 目录"""
    sfm_dir = output_dir / "sfm"
    
    if best_model == "sfm":
        print("  最佳模型已经在 sfm/ 目录中，无需移动")
        return True
    
    # 解析模型路径
    if best_model.startswith("models/"):
        model_index = best_model.split("/")[1]
        source_dir = sfm_dir / "models" / model_index
    else:
        print(f"  未知的模型路径格式: {best_model}")
        return False
    
    if not source_dir.exists():
        print(f"  源目录不存在: {source_dir}")
        return False
    
    print(f"  准备移动 {best_model} 到 sfm/")
    
    if not dry_run:
        # 将原始 sfm 文件备份到空的 models 文件夹中
        backup_folder = backup_sfm_to_models(sfm_dir)
        
        # 移动文件
        filenames = ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]
        success_count = 0
        
        for filename in filenames:
            source_file = source_dir / filename
            target_file = sfm_dir / filename
            
            if source_file.exists():
                try:
                    # 删除现有文件
                    if target_file.exists():
                        target_file.unlink()
                    
                    # 复制文件（使用复制而不是移动，保留原始文件）
                    shutil.copy2(str(source_file), str(target_file))
                    print(f"    ✓ 复制 {filename}")
                    success_count += 1
                except Exception as e:
                    print(f"    ✗ 复制 {filename} 失败: {e}")
            else:
                print(f"    ⚠ 源文件不存在: {filename}")
        
        # 验证移动结果
        final_reconstruction = read_reconstruction_safe(sfm_dir)
        if final_reconstruction:
            final_images = final_reconstruction.num_reg_images()
            print(f"  ✓ 移动完成，最终 sfm/ 目录有 {final_images} 个注册图像")
            return success_count == len(filenames)
        else:
            print(f"  ✗ 移动后无法读取 sfm/ 目录")
            return False
    else:
        print("  (模拟运行，实际未移动文件)")
        return True


def main():
    parser = argparse.ArgumentParser(description="修复 HLOC 重建中的模型选择问题")
    parser.add_argument("--outputs-dir", type=Path, default="YOUR_DATA_PATH-hloc", 
                       help="输出目录路径 (默认: outputs)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="只分析不实际移动文件")
    parser.add_argument("--specific-uuid", type=str, 
                       help="只处理特定的 UUID 文件夹（已废弃，请使用 --target-cases）")
    parser.add_argument("--target-cases", type=str, default=None,
                       help="逗号分隔的目标case ID列表（如果指定，只处理这些case）")
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"错误: 输出目录不存在: {outputs_dir}")
        sys.exit(1)
    
    print(f"扫描输出目录: {outputs_dir}")
    print(f"模式: {'模拟运行' if args.dry_run else '实际执行'}")
    
    # 获取所有输出文件夹
    if args.target_cases:
        # 只处理指定的cases
        target_case_ids = set(args.target_cases.split(','))
        output_folders = [outputs_dir / case_id for case_id in target_case_ids if (outputs_dir / case_id).exists()]
        print(f"目标处理 {len(target_case_ids)} 个指定的case，找到 {len(output_folders)} 个有效文件夹")
    elif args.specific_uuid:
        # 兼容旧参数
        print("警告: --specific-uuid 已废弃，请使用 --target-cases")
        output_folders = [outputs_dir / args.specific_uuid]
        if not output_folders[0].exists():
            print(f"错误: 指定的 UUID 文件夹不存在: {output_folders[0]}")
            sys.exit(1)
    else:
        output_folders = [f for f in outputs_dir.iterdir() if f.is_dir()]
    
    print(f"找到 {len(output_folders)} 个输出文件夹")
    
    # 统计信息
    total_processed = 0
    fixed_count = 0
    error_count = 0
    skip_count = 0
    
    # 处理每个文件夹
    for output_dir in sorted(output_folders):
        try:
            best_model, max_images, models_info = analyze_output_folder(output_dir)
            
            if not best_model:
                print(f"  跳过: 无法分析或没有有效模型")
                print(f"    跳过的文件夹: {output_dir.name}")
                skip_count += 1
                continue
            
            total_processed += 1
            
            # 检查是否需要修复 - 只要有更好的模型就修复
            current_sfm_images = models_info.get("sfm", 0)
            if best_model == "sfm":
                print(f"  ✓ 无需修复: 最佳模型已在 sfm/ 目录")
            elif current_sfm_images >= max_images:
                print(f"  ✓ 无需修复: sfm/ 目录已包含最佳模型 ({current_sfm_images} >= {max_images})")
            else:
                print(f"  需要修复: 当前 sfm/ 有 {current_sfm_images} 图像，最佳模型 {best_model} 有 {max_images} 图像")
                
                if move_model_to_sfm(output_dir, best_model, args.dry_run):
                    print(f"  ✓ 修复成功")
                    fixed_count += 1
                else:
                    print(f"  ✗ 修复失败")
                    error_count += 1
        
        except Exception as e:
            print(f"  ✗ 处理出错: {e}")
            error_count += 1
    
    # 总结
    print(f"\n=== 处理完成 ===")
    print(f"总文件夹数: {len(output_folders)}")
    print(f"跳过: {skip_count}")
    print(f"已处理: {total_processed}")
    print(f"修复成功: {fixed_count}")
    print(f"修复失败: {error_count}")
    
    if args.dry_run:
        print(f"\n这是模拟运行。要实际执行修复，请去掉 --dry_run 参数")


if __name__ == "__main__":
    main()
