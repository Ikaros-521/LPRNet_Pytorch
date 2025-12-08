import os
import random
import shutil
import cv2
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ================= 配置区域 =================

# 输出根目录
OUTPUT_ROOT = "data_win" # 修改一下输出目录名，避免混淆

# 每个类别生成的图片数量
SAMPLES_PER_CLASS_TRAIN = 20
SAMPLES_PER_CLASS_TEST = 5

# 字体路径配置 (请将字体文件放在 fonts 文件夹下)
# Windows下如果没有这些字体，脚本会自动回退到默认字体(虽然丑但能跑)
FONT_DIGITAL = [
    "fonts/digital-7-mono-3.ttf", 
    "fonts/DS-DIGI.TTF",
    "arial.ttf"
]
FONT_CHINESE = [
    "simhei.ttf",  # Windows 自带黑体
    "msyh.ttf"     # Windows 自带微软雅黑
]

# LED 颜色 (RGB)
COLORS = [
    (255, 0, 0),       # Red
    (255, 200, 0),     # Amber
    (255, 255, 0),     # Yellow
    (0, 255, 0),       # Green
    (0, 128, 255),     # Blue-ish (scoreboard LED 蓝色)
    (255, 255, 255)    # White
]

# ================= 标签定义 =================

def get_period_labels(cn_only=False):
    labels = []
    # 中文：第一节到第十节
    labels.extend([f"第{i}节" for i in ["一","二","三","四","五","六","七","八","九","十"]])
    # 中文数字节次：第1-10节
    labels.extend([f"第{i}节" for i in range(1, 11)])
    # 中文加时：加时赛、加时赛1-7、加时一-加时七、加时1-加时7、加时赛一-加时赛七
    cn_nums = ["一","二","三","四","五","六","七"]
    labels.extend(["加时赛"])
    labels.extend([f"加时赛{i}" for i in range(1, 8)])
    labels.extend([f"加时{i}" for i in cn_nums])            # 加时一...
    labels.extend([f"加时{i}" for i in range(1, 8)])        # 加时1...
    labels.extend([f"加时赛{i}" for i in cn_nums])          # 加时赛一...
    if not cn_only:
        # 英文：FIRST, SECOND, THIRD, FOURTH, OT, OVERTIME, OT1-OT7, OVERTIME1-7
        labels.extend(["FIRST", "SECOND", "THIRD", "FOURTH", "OT", "OVERTIME"])
        labels.extend([f"OT{i}" for i in range(1, 8)])
        labels.extend([f"OVERTIME{i}" for i in range(1, 8)])
        # 缩写：1st, 2nd, 3rd, 4th, overtime, ot1-ot7, overtime1-7
        labels.extend(["1st", "2nd", "3rd", "4th", "overtime"])
        labels.extend([f"ot{i}" for i in range(1, 8)])
        labels.extend([f"overtime{i}" for i in range(1, 8)])
    return labels

def get_score_labels():
    # 000-999 (3位数字比分，覆盖所有可能)
    return [f"{i:03d}" for i in range(1000)]

def get_shotclock_labels():
    # 00-24
    return [f"{i:02d}" for i in range(25)]

def get_time_labels():
    labels = set()

    # MM:SS 格式（分秒）
    for m in range(13):  # 0-12分钟
        for s in range(0, 60, 5):  # 每5秒一个，覆盖常见时间
            labels.add(f"{m:02d}:{s:02d}")

    def add_ms(prefix: str):
        """为给定前缀添加 1/2/3 位毫秒"""
        for width in (1, 2, 3):
            if width == 1:
                values = range(10)  # 0-9
            elif width == 2:
                values = list(range(0, 100, 10)) + [11, 33, 59, 88]
            else:
                values = list(range(0, 1000, 100)) + [123, 456, 789]
            for ms in values:
                labels.add(f"{prefix}.{ms:0{width}d}")

    # MM:SS.ms / .mm / .mmm
    for m in range(13):
        for s in [0, 10, 20, 30, 40, 50, 59]:  # 常见关键秒
            add_ms(f"{m:02d}:{s:02d}")

    # SS.ms / .mm / .mmm 仅秒部分（最后一分钟）
    for s in range(60):
        add_ms(f"{s:02d}")

    return list(labels)

# ================= 图像处理核心 =================

def load_font(text, size):
    is_chinese = any(u'\u4e00' <= char <= u'\u9fff' for char in text)
    font_list = FONT_CHINESE if is_chinese else FONT_DIGITAL
    
    # 尝试加载字体，失败则用默认
    for f_path in font_list:
        try:
            # 优先尝试在当前目录查找，或者在系统字体目录查找
            return ImageFont.truetype(f_path, size)
        except:
            continue
    
    # 如果都失败，尝试直接加载系统字体名 (Windows特有)
    try:
        if is_chinese:
            return ImageFont.truetype("simhei.ttf", size)
        else:
            return ImageFont.truetype("arial.ttf", size)
    except:
        pass
        
    print(f"Warning: Using default PIL font for '{text}'")
    return ImageFont.load_default()

def apply_augmentations(cv_img):
    rows, cols = cv_img.shape[:2]

    # 1. 扫描线（模拟LED行噪声）
    for i in range(0, rows, 3):
        cv_img[i, :] = (cv_img[i, :] * 0.7).astype(np.uint8)

    # 2. 轻微旋转/倾斜
    if random.random() > 0.5:
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
        cv_img = cv2.warpAffine(cv_img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))

    # 3. 轻微透视畸变
    if random.random() > 0.6:
        shift = lambda : random.uniform(-3, 3)
        src = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
        dst = np.float32([[shift(),shift()],
                          [cols+shift(), shift()],
                          [shift(), rows+shift()],
                          [cols+shift(), rows+shift()]])
        M = cv2.getPerspectiveTransform(src, dst)
        cv_img = cv2.warpPerspective(cv_img, M, (cols, rows), borderValue=(0,0,0))

    # 4. 模糊
    if random.random() > 0.3:
        ksize = random.choice([3, 5])
        cv_img = cv2.GaussianBlur(cv_img, (ksize, ksize), 0)

    # 5. 噪点（轻微即可）
    if random.random() > 0.25:
        noise = np.random.normal(0, 3, cv_img.shape).astype(np.int16)
        cv_img = cv_img.astype(np.int16) + noise
        cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)

    # 6. 荧光眩光/光晕
    if random.random() > 0.6:
        glow = np.zeros_like(cv_img)
        cx, cy = random.randint(cols//4, 3*cols//4), random.randint(rows//4, 3*rows//4)
        radius = random.randint(rows//3, rows)
        cv2.circle(glow, (cx, cy), radius, (255,255,255), -1)
        glow = cv2.GaussianBlur(glow, (0,0), sigmaX=random.uniform(5,12))
        alpha = random.uniform(0.15, 0.35)
        cv_img = cv2.addWeighted(cv_img, 1.0, glow, alpha, 0)

    return cv_img

def save_image_unicode(img_cv, save_path, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img_cv)
    if not ok:
        raise ValueError(f"Failed to encode image for {save_path}")
    buf.tofile(save_path)


def generate_image(text, save_path):
    img_h = 64
    font_size = random.randint(40, 50)
    kerning = random.randint(-5, 2) # 粘连控制
    
    # 主体字体与毫秒字体（毫秒更小）
    font_main = load_font(text, font_size)
    ms_font_size = max(20, int(font_size * random.uniform(0.55, 0.8)))
    font_ms = load_font(text, ms_font_size)

    temp_w = int(font_size * 0.85 * len(text)) + 60
    img_pil = Image.new('RGB', (temp_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img_pil)
    color = random.choice(COLORS)
    
    current_x = 10
    y_main = (img_h - font_size) // 2 - 5
    y_ms = y_main + (font_size - ms_font_size) // 2  # 让毫秒稍微靠上保持对齐

    after_dot = False
    for char in text:
        font = font_ms if after_dot or char == '.' else font_main

        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), char, font=font)
            char_w = bbox[2] - bbox[0]
        else:
            char_w, _ = draw.textsize(char, font=font)

        y_pos = y_ms if (after_dot or char == '.') else y_main
        draw.text((current_x, y_pos), char, font=font, fill=color)
        current_x += char_w + kerning

        if char == '.':
            after_dot = True
    
    # Crop
    bbox = img_pil.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        img_pil = img_pil.crop((left-5, 0, right+5, img_h))
    
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_cv = apply_augmentations(img_cv)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image_unicode(img_cv, save_path, ".jpg")

# ================= Windows 路径清洗 =================

def sanitize_filename(text):
    """
    将 Windows 非法字符转换为安全字符
    例如: '34:56' -> '34_56'
    """
    invalid_chars = '<>:"/\\|?*'
    safe_text = text
    for char in invalid_chars:
        safe_text = safe_text.replace(char, '_') # 统一替换为下划线
    return safe_text

# ================= CLI 参数 =================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_cn", action="store_true",
                        help="只生成中文标签（仅节次相关），不生成数字比分/时间/24秒")
    parser.add_argument("--cn_period_only", action="store_true",
                        help="节次标签仅使用中文变体，去除英文/缩写")
    parser.add_argument("--train_samples", type=int, default=SAMPLES_PER_CLASS_TRAIN,
                        help="每类训练样本数")
    parser.add_argument("--test_samples", type=int, default=SAMPLES_PER_CLASS_TEST,
                        help="每类测试样本数")
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT,
                        help="输出根目录")
    parser.add_argument("--clean", action="store_true",
                        help="生成前清空输出目录")
    return parser.parse_args()

# ================= 主程序 =================

def main():
    args = parse_args()

    # 准备任务
    period_labels = get_period_labels(cn_only=(args.cn_period_only or args.only_cn))
    if args.only_cn:
        tasks = [period_labels]
    else:
        score_labels = get_score_labels()
        shotclock_labels = get_shotclock_labels()
        time_labels = get_time_labels()
        tasks = [score_labels, shotclock_labels, time_labels, period_labels]

    all_labels = set()
    for t in tasks:
        all_labels.update(t)
    
    # 打印统计信息
    print("=" * 60)
    print("数据集生成统计:")
    if not args.only_cn:
        print(f"  比分标签: {len(tasks[0])} 个 (000-999)")
        print(f"  24秒倒计时标签: {len(tasks[1])} 个 (00-24)")
        print(f"  时间标签: {len(tasks[2])} 个 (MM:SS, MM:SS.ms, SS.ms)")
    print(f"  节次标签: {len(period_labels)} 个")
    print(f"  总唯一标签数: {len(all_labels)} 个")
    print("=" * 60)
    
    # 可选清理输出目录
    if args.clean and os.path.exists(args.output_root):
        shutil.rmtree(args.output_root, ignore_errors=True)

    # 生成训练集，直接写入对应标签文件夹，文件名用数字
    print("Generating train set...")
    for label_text in all_labels:
        safe_folder_name = sanitize_filename(label_text)
        folder_path = os.path.join(args.output_root, 'train', safe_folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for i in range(args.train_samples):
            file_name = f"{i:03d}.jpg"
            full_path = os.path.join(folder_path, file_name)
            generate_image(label_text, full_path)
    print("Finished train.")

    # 生成测试集：从训练集中拷贝部分样本
    print("Generating test set (copy from train)...")
    for label_text in all_labels:
        safe_folder_name = sanitize_filename(label_text)
        src_folder = os.path.join(args.output_root, 'train', safe_folder_name)
        dst_folder = os.path.join(args.output_root, 'test', safe_folder_name)
        os.makedirs(dst_folder, exist_ok=True)
        files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith('.jpg')])
        take_n = min(args.test_samples, len(files))
        for f in files[:take_n]:
            shutil.copy2(os.path.join(src_folder, f), os.path.join(dst_folder, f))
    print("Finished test.")

if __name__ == "__main__":
    main()