import argparse
import os
import numpy as np
import cv2
import torch

from data.load_data import CHARS, imread_unicode
from model.LPRNet import build_lprnet


def letterbox(img, img_size):
    """Keep aspect ratio; pad to target size."""
    tgt_w, tgt_h = img_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image size")

    scale = tgt_h / h
    new_w = int(w * scale)
    new_h = tgt_h
    if new_w > tgt_w:
        scale = tgt_w / w
        new_w = tgt_w
        new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((tgt_h, tgt_w, 3), dtype=resized.dtype)
    x0 = (tgt_w - new_w) // 2
    y0 = (tgt_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w, :] = resized
    return canvas


def preprocess(img, img_size):
    img = cv2.resize(img, tuple(img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32")
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def greedy_decode(logits):
    # logits: N x C x W
    probs = logits.softmax(1)
    top = probs.argmax(1)[0].cpu().numpy()  # W
    blank = len(CHARS) - 1
    out = []
    prev = blank
    for c in top:
        if c != prev and c != blank:
            out.append(CHARS[c])
        prev = c
    return "".join(out)


def main():
    # python infer.py --image data_win/test/第一节/000.jpg \
    #             --pretrained_model weights/Scoreboard__iteration_16000.pth \
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
    text = greedy_decode(logits)

    print(f"[Result] {os.path.basename(args.image)} -> {text}")


if __name__ == "__main__":
    main()

