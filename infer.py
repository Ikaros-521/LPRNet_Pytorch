import argparse
import os
import numpy as np
import cv2
import torch

from data.load_data import CHARS, imread_unicode
from model.LPRNet import build_lprnet


def preprocess(img, img_size):
    """直接拉伸到目标尺寸（与训练保持一致）"""
    img = cv2.resize(img, tuple(img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32")
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def greedy_decode(logits, max_len=None):
    # logits: N x C x W
    probs = logits.softmax(1)
    top = probs.argmax(1)[0].cpu().numpy()  # W
    blank = len(CHARS) - 1
    out = []
    prev = blank
    for c in top:
        if c != prev and c != blank:
            out.append(CHARS[c])
            # 可选：截断到最大长度，避免解码出过长结果
            if max_len is not None and len(out) >= max_len:
                break
        prev = c
    return "".join(out)


def main():
    # python infer.py --image data_win/test/FIRST/000.jpg \
    #             --pretrained_model weights/Final_Scoreboard_model.pth \
    #             --img_size 94 24 \
    #             --max_len 12
    parser = argparse.ArgumentParser(description="Single image inference for scoreboard")
    parser.add_argument("--image", required=True, help="path to image")
    parser.add_argument("--pretrained_model", required=True, help="path to weights")
    parser.add_argument("--img_size", default=[94, 24], nargs=2, type=int, help="model input size [w h]")
    parser.add_argument("--max_len", default=12, type=int, help="max length (for model build)")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    net = build_lprnet(lpr_max_len=args.max_len, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.to(device)
    net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    net.eval()

    img = imread_unicode(args.image)
    inp = preprocess(img, img_size=args.img_size).to(device)

    with torch.no_grad():
        logits = net(inp)  # N x C x W
    text = greedy_decode(logits, max_len=args.max_len)

    print(f"[Result] {os.path.basename(args.image)} -> {text}")


if __name__ == "__main__":
    main()

