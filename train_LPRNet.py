# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for Basketball Scoreboard Recognition.
基于LPRNet架构改造，用于识别篮球计分板信息：
- 比分（3位数字）
- 24秒倒计时（2位数字）
- 倒计时（时分秒、时分秒毫秒）
- 节次（第一节、第1节、FIRST、1st等）
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os

def sparse_tuple_for_ctc(T_length, lengths):
    """
    为CTC loss准备输入长度和目标长度
    确保返回的是整数元组
    """
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        # 确保T_length和ch都是整数
        input_lengths.append(int(T_length))
        target_lengths.append(int(ch))

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train basketball scoreboard recognition net')
    # 训练轮数
    parser.add_argument('--max_epoch', type=int, default=25, help='epoch to train the network')
    # 默认输入分辨率（与模型设计匹配）
    parser.add_argument('--img_size', nargs=2, type=int, default=[94, 24], help='the image size [width, height]')
    # 训练图片目录
    parser.add_argument('--train_img_dirs', default="./data/train", help='the train images path')
    # 测试图片目录
    parser.add_argument('--test_img_dirs', default="./data/test", help='the test images path')
    # Dropout率
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate.')
    # 学习率（降低默认值，避免梯度爆炸）
    parser.add_argument('--learning_rate', type=float, default=0.001, help='base value of learning rate.')
    # 最大文本长度（比分3位，24秒2位，时间MM:SS为5位）
    parser.add_argument('--max_len', type=int, default=5, help='maximum text length (比分最长3位，时间MM:SS最长5位)')
    # 训练批次大小
    parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size.')
    # 测试批次大小
    parser.add_argument('--test_batch_size', type=int, default=120, help='testing batch size.')
    # 训练或测试阶段标志
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    # 数据加载线程数
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    # 是否使用GPU训练
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    # 恢复训练的迭代次数
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    # 保存模型状态字典的间隔
    parser.add_argument('--save_interval', default=4000, type=int, help='interval for save model state dict')
    # 评估模型的间隔
    parser.add_argument('--test_interval', default=1000, type=int, help='interval for evaluate')
    # 动量
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # 权重衰减
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    # 学习率衰减计划（更长训练，更多衰减节点）
    parser.add_argument('--lr_schedule', default=[8, 12, 16, 20, 23], help='schedule for learning rate.')
    # 保存模型状态字典的文件夹
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # 预训练模型
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()
    # 保持向后兼容
    if not hasattr(args, 'lpr_max_len'):
        args.lpr_max_len = args.max_len

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int64)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    args = get_parser()

    # 使用max_len参数，如果没有则使用lpr_max_len（向后兼容）
    max_len = getattr(args, 'max_len', args.lpr_max_len)
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    print("=" * 60)
    print("模型配置信息:")
    print(f"  字符集: {CHARS}")
    print(f"  字符集长度: {len(CHARS)}")
    print(f"  CTC Blank索引: {len(CHARS)-1} (字符: '{CHARS[len(CHARS)-1]}')")
    print(f"  最大标签长度: {max_len}")
    print("=" * 60)
    
    lprnet = build_lprnet(lpr_max_len=max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        print(f"⚠️ 警告: 正在加载预训练模型: {args.pretrained_model}")
        print("⚠️ 如果预训练模型是用不同的字符集训练的，可能会导致问题！")
        try:
            lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device))
            print("load pretrained model successful!")
        except Exception as e:
            print(f"❌ 加载预训练模型失败: {e}")
            print("建议: 删除 --pretrained_model 参数，从头开始训练")
            raise
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    max_len = getattr(args, 'max_len', args.lpr_max_len)
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, max_len)
    
    # 打印数据集信息
    print(f"\n数据集信息:")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")
    if len(train_dataset) > 0:
        # 检查第一个样本的标签
        _, first_label, first_length = train_dataset[0]
        first_label_str = ''.join([CHARS[idx] for idx in first_label])
        print(f"  第一个训练样本标签: {first_label} -> '{first_label_str}' (长度: {first_length})")
        # 检查标签索引范围
        all_labels = []
        for i in range(min(100, len(train_dataset))):  # 只检查前100个
            _, label, _ = train_dataset[i]
            all_labels.extend(label)
        if all_labels:
            print(f"  标签索引范围: [{min(all_labels)}, {max(all_labels)}] (期望: [0, {len(CHARS)-1}])")
            invalid = [idx for idx in all_labels if idx < 0 or idx >= len(CHARS)]
            if invalid:
                print(f"  ⚠️ 警告: 发现 {len(invalid)} 个无效标签索引!")
    print()

    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size

    # 使用 log_probs (log_softmax) 作为输入，zero_infinity=True 避免无穷大导致 NaN
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean', zero_infinity=True)

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'Scoreboard_' + '_iteration_' + repr(iteration) + '.pth')

        if (iteration + 1) % args.test_interval == 0:
            Greedy_Decode_Eval(lprnet, test_dataset, args)
            # lprnet.train() # should be switch to train mode

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        
        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # 动态获取实际的时间维度（宽度）
        T_length = log_probs.shape[0]
        
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)
        
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)
        # backprop
        optimizer.zero_grad()
        # 确保input_lengths和target_lengths是整数元组
        input_lengths_int = tuple(int(x) for x in input_lengths)
        target_lengths_int = tuple(int(x) for x in target_lengths)
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths_int, target_lengths=target_lengths_int)
        
        # 检查loss是否为nan或inf
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() == np.inf:
            print(f"Warning: Invalid loss detected (nan/inf) at iteration {iteration}")
            print(f"  Loss value: {loss.item()}")
            print(f"  Log_probs shape: {log_probs.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels range: [{labels.min().item()}, {labels.max().item()}]")
            print(f"  Expected label range: [0, {len(CHARS)-1}]")
            print(f"  Input lengths: {input_lengths}")
            print(f"  Target lengths: {target_lengths}")
            print(f"  Class num: {len(CHARS)}")
            # 检查标签中是否有无效索引
            invalid_labels = labels[(labels < 0) | (labels >= len(CHARS))]
            if len(invalid_labels) > 0:
                print(f"  ⚠️ Found {len(invalid_labels)} invalid label indices!")
                print(f"  Invalid indices: {invalid_labels.unique().cpu().numpy()}")
            continue
        loss.backward()

        # 梯度裁剪，避免梯度爆炸
        torch.nn.utils.clip_grad_norm_(lprnet.parameters(), max_norm=5.0)

        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        if iteration % 20 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))
    # final test
    print("Final test Accuracy:")
    Greedy_Decode_Eval(lprnet, test_dataset, args)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_Scoreboard_model.pth')

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        # targets 的长度不一致，保持为 list/可变长数组，避免 np.array 自动填充失败
        targets = [el.numpy() for el in targets]

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            # 截断到最大长度，避免输出过长序列
            if len(no_repeat_blank_label) > args.max_len:
                no_repeat_blank_label = no_repeat_blank_label[:args.max_len]
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))


if __name__ == "__main__":
    train()
