#!/usr/bin/env python3
"""
视频帧采样脚本
支持按指定帧率从视频中提取帧
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class VideoFrameSampler:
    """视频帧采样器"""
    
    def __init__(self, video_path: str, output_dir: str = None):
        """
        初始化采样器
        
        Args:
            video_path: 视频文件路径或视频文件夹路径
            output_dir: 输出目录，默认为 video_path/frames
        """
        self.video_path = Path(video_path)
        
        if output_dir is None:
            if self.video_path.is_file():
                # 如果是单个文件，在文件所在目录创建frames文件夹
                self.base_output_dir = self.video_path.parent / "frames"
            else:
                # 如果是文件夹，在该文件夹下创建frames文件夹
                self.base_output_dir = self.video_path / "frames"
        else:
            self.base_output_dir = Path(output_dir)
        
        # 创建基础输出目录
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于线程安全打印的锁
        self.print_lock = Lock()
        
    def get_video_info(self, video_path: str) -> Tuple[int, float, int, int]:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            (总帧数, 帧率, 宽度, 高度)
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return total_frames, fps, width, height
    
    def sample_frames_by_fps(self, video_path: str, target_fps: float, 
                           max_frames: Optional[int] = None) -> List[str]:
        """
        按指定帧率采样视频帧
        
        Args:
            video_path: 视频文件路径
            target_fps: 目标采样帧率
            max_frames: 最大采样帧数，None表示不限制
            
        Returns:
            采样帧的文件路径列表
        """
        # 获取视频信息
        total_frames, original_fps, width, height = self.get_video_info(video_path)
        
        print(f"处理视频: {Path(video_path).name}")
        print(f"原始帧率: {original_fps:.2f} fps")
        print(f"总帧数: {total_frames}")
        print(f"分辨率: {width}x{height}")
        print(f"目标采样帧率: {target_fps} fps")
        
        # 计算采样间隔
        if target_fps >= original_fps:
            # 如果目标帧率大于等于原始帧率，采样所有帧
            frame_interval = 1
            print("目标帧率大于等于原始帧率，将采样所有帧")
        else:
            # 计算每隔多少帧采样一次
            frame_interval = int(original_fps / target_fps)
            print(f"每隔 {frame_interval} 帧采样一次")
        
        # 计算预期采样帧数
        expected_samples = total_frames // frame_interval
        if max_frames is not None and expected_samples > max_frames:
            # 如果超过最大限制，重新计算间隔
            frame_interval = total_frames // max_frames
            expected_samples = max_frames
            print(f"限制最大帧数为 {max_frames}，调整采样间隔为 {frame_interval}")
        
        print(f"预计采样 {expected_samples} 帧")
        
        # 开始采样
        cap = cv2.VideoCapture(video_path)
        video_name = Path(video_path).stem
        
        # 为当前视频创建专门的输出文件夹
        video_output_dir = self.base_output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 按间隔采样
            if frame_count % frame_interval == 0:
                if max_frames is not None and saved_count >= max_frames:
                    break
                    
                # 保存帧
                frame_filename = f"frame_{saved_count:06d}.png"
                frame_path = video_output_dir / frame_filename
                
                success = cv2.imwrite(str(frame_path), frame)
                if success:
                    saved_frames.append(str(frame_path))
                    saved_count += 1
                    
                    if saved_count % 100 == 0:
                        print(f"已保存 {saved_count} 帧...")
                        
            frame_count += 1
            
        cap.release()
        
        print(f"完成！共保存 {saved_count} 帧到 {video_output_dir}")
        return saved_frames
    
    def sample_frames_by_interval(self, video_path: str, interval_seconds: float,
                                max_frames: Optional[int] = None) -> List[str]:
        """
        按时间间隔采样视频帧
        
        Args:
            video_path: 视频文件路径
            interval_seconds: 采样时间间隔（秒）
            max_frames: 最大采样帧数，None表示不限制
            
        Returns:
            采样帧的文件路径列表
        """
        # 获取视频信息
        total_frames, fps, width, height = self.get_video_info(video_path)
        duration = total_frames / fps
        
        print(f"处理视频: {Path(video_path).name}")
        print(f"视频时长: {duration:.2f} 秒")
        print(f"采样间隔: {interval_seconds} 秒")
        
        # 计算采样时间点
        sample_times = []
        current_time = 0
        while current_time < duration:
            sample_times.append(current_time)
            current_time += interval_seconds
            if max_frames is not None and len(sample_times) >= max_frames:
                break
        
        print(f"预计采样 {len(sample_times)} 帧")
        
        # 开始采样
        cap = cv2.VideoCapture(video_path)
        video_name = Path(video_path).stem
        
        # 为当前视频创建专门的输出文件夹
        video_output_dir = self.base_output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        
        for i, sample_time in enumerate(sample_times):
            # 跳转到指定时间
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            
            ret, frame = cap.read()
            if not ret:
                break
                
            # 保存帧
            frame_filename = f"time_{sample_time:.2f}s_frame_{i:06d}.png"
            frame_path = video_output_dir / frame_filename
            
            success = cv2.imwrite(str(frame_path), frame)
            if success:
                saved_frames.append(str(frame_path))
                
                if (i + 1) % 50 == 0:
                    print(f"已保存 {i + 1} 帧...")
                    
        cap.release()
        
        print(f"完成！共保存 {len(saved_frames)} 帧到 {video_output_dir}")
        return saved_frames
    
    def _process_single_video(self, video_file: Path, video_idx: int, total_videos: int,
                             target_fps: float = None, interval_seconds: float = None,
                             max_frames: Optional[int] = None) -> Tuple[str, List[str]]:
        """
        处理单个视频文件（用于多线程）
        
        Args:
            video_file: 视频文件路径
            video_idx: 视频索引
            total_videos: 视频总数
            target_fps: 目标采样帧率
            interval_seconds: 采样时间间隔
            max_frames: 最大采样帧数
            
        Returns:
            (视频路径, 帧路径列表)
        """
        with self.print_lock:
            print(f"\n--- 处理第 {video_idx}/{total_videos} 个视频 ---")
        
        try:
            if target_fps is not None:
                frames = self.sample_frames_by_fps(str(video_file), target_fps, max_frames)
            elif interval_seconds is not None:
                frames = self.sample_frames_by_interval(str(video_file), interval_seconds, max_frames)
            else:
                raise ValueError("必须指定 target_fps 或 interval_seconds")
            
            return (str(video_file), frames)
            
        except Exception as e:
            with self.print_lock:
                print(f"处理视频 {video_file.name} 时出错: {e}")
            return (str(video_file), [])
    
    def process_directory(self, target_fps: float = None, interval_seconds: float = None,
                         max_frames: Optional[int] = None, video_extensions: List[str] = None,
                         max_workers: int = 4) -> dict:
        """
        批量处理目录中的所有视频文件（多线程并行）
        
        Args:
            target_fps: 目标采样帧率（与interval_seconds二选一）
            interval_seconds: 采样时间间隔（与target_fps二选一）
            max_frames: 每个视频的最大采样帧数
            video_extensions: 支持的视频文件扩展名
            max_workers: 最大并行工作线程数，默认为4
            
        Returns:
            处理结果字典 {video_path: [frame_paths]}
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if not self.video_path.is_dir():
            raise ValueError(f"{self.video_path} 不是一个有效的目录")
        
        # 查找所有视频文件
        video_files = []
        for ext in video_extensions:
            video_files.extend(self.video_path.glob(f"*{ext}"))
        
        print(f"在 {self.video_path} 中找到 {len(video_files)} 个视频文件")
        print(f"使用 {max_workers} 个线程并行处理")
        
        if not video_files:
            print("未找到任何视频文件")
            return {}
        
        results = {}
        
        # 使用线程池并行处理视频
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_video = {}
            for i, video_file in enumerate(video_files):
                future = executor.submit(
                    self._process_single_video,
                    video_file,
                    i + 1,
                    len(video_files),
                    target_fps,
                    interval_seconds,
                    max_frames
                )
                future_to_video[future] = video_file
            
            # 收集结果
            for future in as_completed(future_to_video):
                video_path, frames = future.result()
                results[video_path] = frames
        
        return results


def main():
    parser = argparse.ArgumentParser(description="视频帧采样工具")
    parser.add_argument("--input_path", default='YOUR_DATA_PATH', help="输入视频文件或视频文件夹路径")
    parser.add_argument("--output",     default='YOUR_DATA_PATH-frames_4fps', help="输出目录")
    parser.add_argument("--fps", type=float, default=4, help="目标采样帧率")
    parser.add_argument("--interval", type=float, help="采样时间间隔（秒）")
    parser.add_argument("--max-frames", type=int, help="每个视频的最大采样帧数")
    parser.add_argument("--max-workers", type=int, default=32, help="并行处理的最大线程数（默认：4）")
    parser.add_argument("--extensions", nargs="+", 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
                       help="支持的视频文件扩展名")
    
    args = parser.parse_args()
    
    if args.fps is None and args.interval is None:
        parser.error("必须指定 --fps 或 --interval 参数")
    
    if args.fps is not None and args.interval is not None:
        parser.error("--fps 和 --interval 参数不能同时使用")
    
    # 创建采样器
    sampler = VideoFrameSampler(args.input_path, args.output)
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # 处理单个文件
        print("处理单个视频文件...")
        if args.fps is not None:
            frames = sampler.sample_frames_by_fps(args.input_path, args.fps, args.max_frames)
        else:
            frames = sampler.sample_frames_by_interval(args.input_path, args.interval, args.max_frames)
        
        print(f"\n总结: 成功采样 {len(frames)} 帧")
        
    elif input_path.is_dir():
        # 处理目录
        print("批量处理视频文件...")
        results = sampler.process_directory(
            target_fps=args.fps,
            interval_seconds=args.interval,
            max_frames=args.max_frames,
            video_extensions=args.extensions,
            max_workers=args.max_workers
        )
        
        # 打印总结
        total_frames = sum(len(frames) for frames in results.values())
        successful_videos = sum(1 for frames in results.values() if frames)
        
        print(f"\n总结:")
        print(f"处理视频数量: {len(results)}")
        print(f"成功处理: {successful_videos}")
        print(f"总采样帧数: {total_frames}")
        
    else:
        print(f"错误: {args.input_path} 不是有效的文件或目录")


if __name__ == "__main__":
    main()
