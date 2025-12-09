# -*- coding: utf-8 -*-
"""
批量推理验证程序
对测试集或训练集进行批量推理，比对结果，统计正确率，记录失败案例
"""

import argparse
import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from data.load_data import CHARS, CHARS_DICT, ScoreboardDataLoader, convert_folder_name_to_label, imread_unicode
from model.LPRNet import build_lprnet


def collate_fn(batch):
    """自定义collate函数，处理变长标签"""
    imgs = []
    labels = []
    lengths = []
    for img, label, length in batch:
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.array(labels, dtype=np.int64)
    imgs = torch.stack(imgs, 0)
    return imgs, labels, lengths


def greedy_decode(logits, max_len=None, return_confidence=False):
    """
    贪婪解码：将模型输出转换为文本
    logits: N x C x W (batch_size x num_classes x width)
    return_confidence: 是否返回置信度
    返回: 文本列表，或 (文本列表, 置信度列表)
    """
    probs = torch.softmax(logits, dim=1)  # N x C x W
    blank = len(CHARS) - 1
    
    results = []
    confidences_list = []
    
    for i in range(logits.shape[0]):
        # 获取每个时间步的最大概率类别和概率值
        top = probs[i].argmax(dim=0).cpu().numpy()  # W
        top_probs = probs[i].max(dim=0)[0].cpu().numpy()  # W，每个位置的最大概率
        
        # 去重和去空白
        out = []
        confidences = []
        prev = blank
        for idx, c in enumerate(top):
            if c != prev and c != blank:
                out.append(CHARS[c])
                confidences.append(float(top_probs[idx]))  # 记录该字符位置的置信度
                if max_len is not None and len(out) >= max_len:
                    break
            prev = c
        
        results.append("".join(out))
        if return_confidence:
            # 计算平均置信度和最小置信度
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            min_conf = float(np.min(confidences)) if confidences else 0.0
            confidences_list.append((avg_conf, min_conf))
    
    if return_confidence:
        return results, confidences_list
    else:
        return results


def label_to_string(label_indices):
    """将标签索引列表转换为字符串"""
    return "".join([CHARS[idx] for idx in label_indices if idx < len(CHARS)])


def validate_dataset(model, dataset, args, device):
    """
    验证数据集，返回统计结果和失败案例
    """
    model.eval()
    
    # 创建数据加载器（不 shuffle，保持顺序一致）
    # 注意：如果使用多线程（num_workers > 0），图片路径索引可能不准确
    # 建议使用 num_workers=0 或确保数据集顺序与 DataLoader 顺序一致
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    total = 0
    correct = 0
    failed_cases = []
    
    # 按标签统计
    label_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # 全局索引计数器
    global_idx = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels, lengths) in enumerate(tqdm(dataloader, desc="验证中")):
            # 准备真实标签
            start = 0
            true_labels = []
            for length in lengths:
                label = labels[start:start+length]  # labels 已经是 numpy array，不需要 .cpu().numpy()
                true_labels.append(label_to_string(label))
                start += length
            
            # 推理
            if args.cuda:
                images = images.cuda()
            else:
                images = images
            
            logits = model(images)  # N x C x W
            pred_labels, pred_confidences = greedy_decode(logits, max_len=args.max_len, return_confidence=True)
            
            # 比对结果
            batch_size = len(true_labels)
            for i in range(batch_size):
                total += 1
                true_label = true_labels[i]
                pred_label = pred_labels[i]
                avg_conf, min_conf = pred_confidences[i]
                
                # 统计该标签的准确率
                label_stats[true_label]['total'] += 1
                
                # 比对（精确匹配）
                is_correct = (true_label == pred_label)
                if is_correct:
                    correct += 1
                    label_stats[true_label]['correct'] += 1
                else:
                    # 记录失败案例（使用全局索引获取对应的图片路径）
                    if global_idx < len(dataset.img_paths):
                        img_path = dataset.img_paths[global_idx]
                        failed_cases.append({
                            'image_path': img_path,
                            'true_label': true_label,
                            'pred_label': pred_label,
                            'confidence_avg': avg_conf,
                            'confidence_min': min_conf
                        })
                
                global_idx += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'failed_cases': failed_cases,
        'label_stats': dict(label_stats)
    }


def main():
    parser = argparse.ArgumentParser(description='批量推理验证程序')
    parser.add_argument('--dataset_dir', required=True, help='数据集目录（训练集或测试集）')
    parser.add_argument('--pretrained_model', required=True, help='模型权重路径')
    parser.add_argument('--img_size', nargs=2, type=int, default=[94, 24], help='模型输入尺寸 [width, height]')
    parser.add_argument('--max_len', type=int, default=5, help='最大文本长度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--cuda', action='store_true', help='使用GPU')
    parser.add_argument('--output_dir', default='./validation_results', help='结果输出目录')
    parser.add_argument('--save_failed_images', action='store_true', help='是否保存失败图片的副本')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.pretrained_model}")
    net = build_lprnet(lpr_max_len=args.max_len, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    net.to(device)
    net.eval()
    
    # 加载数据集
    print(f"加载数据集: {args.dataset_dir}")
    dataset = ScoreboardDataLoader(
        img_dir=[args.dataset_dir],
        imgSize=args.img_size,
        max_len=args.max_len,
        PreprocFun=None
    )
    
    if len(dataset) == 0:
        print(f"错误: 数据集目录 {args.dataset_dir} 中没有找到有效图片！")
        return
    
    print(f"数据集大小: {len(dataset)} 张图片")
    print(f"唯一标签数: {len(set([tuple(l) for l in dataset.labels]))}")
    print()
    
    # 执行验证
    results = validate_dataset(net, dataset, args, device)
    
    # 打印统计结果
    print("\n" + "="*60)
    print("验证结果统计")
    print("="*60)
    print(f"总图片数: {results['total']}")
    print(f"正确识别: {results['correct']}")
    print(f"错误识别: {results['total'] - results['correct']}")
    print(f"准确率: {results['accuracy']:.2f}%")
    print()
    
    # 按标签统计
    if results['label_stats']:
        print("按标签统计（前20个）:")
        print("-" * 60)
        sorted_labels = sorted(
            results['label_stats'].items(), 
            key=lambda x: x[1]['total'], 
            reverse=True
        )[:20]
        
        for label, stats in sorted_labels:
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
            print(f"  {label:15s} | 总数: {stats['total']:4d} | 正确: {stats['correct']:4d} | 准确率: {acc:6.2f}%")
        print()
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存失败案例
    if results['failed_cases']:
        failed_file = os.path.join(args.output_dir, 'failed_cases.json')
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(results['failed_cases'], f, ensure_ascii=False, indent=2)
        print(f"失败案例已保存到: {failed_file}")
        print(f"失败案例数: {len(results['failed_cases'])}")
        
        # 统计失败案例的置信度信息
        if results['failed_cases'] and 'confidence_avg' in results['failed_cases'][0]:
            conf_avgs = [case['confidence_avg'] for case in results['failed_cases']]
            conf_mins = [case['confidence_min'] for case in results['failed_cases']]
            print(f"失败案例置信度统计:")
            print(f"  平均置信度: {np.mean(conf_avgs):.4f} (avg), {np.min(conf_avgs):.4f} (min), {np.max(conf_avgs):.4f} (max)")
            print(f"  最小置信度: {np.mean(conf_mins):.4f} (avg), {np.min(conf_mins):.4f} (min), {np.max(conf_mins):.4f} (max)")
        
        # 如果启用，复制失败图片
        if args.save_failed_images:
            failed_img_dir = os.path.join(args.output_dir, 'failed_images')
            os.makedirs(failed_img_dir, exist_ok=True)
            
            print(f"正在复制失败图片到: {failed_img_dir}")
            for idx, case in enumerate(tqdm(results['failed_cases'], desc="复制图片")):
                img_path = case['image_path']
                if os.path.exists(img_path):
                    # 生成新文件名：索引_真实标签_预测标签_原文件名
                    base_name = os.path.basename(img_path)
                    name, ext = os.path.splitext(base_name)
                    new_name = f"{idx:04d}_{case['true_label']}_{case['pred_label']}_{name}{ext}"
                    new_path = os.path.join(failed_img_dir, new_name)
                    
                    # 读取并保存（处理Unicode路径）
                    try:
                        img = imread_unicode(img_path)
                        cv2.imencode(ext, img)[1].tofile(new_path)
                    except Exception as e:
                        print(f"警告: 无法复制图片 {img_path}: {e}")
    else:
        print("所有图片识别正确！")
    
    # 保存完整统计结果
    summary_file = os.path.join(args.output_dir, 'summary.json')
    summary = {
        'total': results['total'],
        'correct': results['correct'],
        'accuracy': results['accuracy'],
        'failed_count': len(results['failed_cases']),
        'label_stats': results['label_stats']
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"统计摘要已保存到: {summary_file}")
    
    print("\n验证完成！")


if __name__ == "__main__":
    # python batch_validate.py --dataset_dir data_win/test --pretrained_model weights/Final_Scoreboard_model.pth --cuda
    main()

