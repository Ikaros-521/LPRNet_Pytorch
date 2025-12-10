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
SAMPLES_PER_CLASS_TRAIN = 100
# 每类测试样本数
SAMPLES_PER_CLASS_TEST = 25

# 字体路径配置 (请将字体文件放在 fonts 文件夹下)
# Windows下如果没有这些字体，脚本会自动回退到默认字体(虽然丑但能跑)
#
# 字体类型说明：
# 1. FONT_NORMAL: 正常数字字体（Arial, Times New Roman, Verdana 等系统常见字体）
# 2. FONT_DIGITAL: 数字显示字体（7段数码管风格，如 digital-7, DS-DIGI 等）
# 3. FONT_DOT_MATRIX: 点阵字体（由点阵组成的数字，模拟LED点阵屏效果）
# 4. FONT_LED_DOT: LED灯珠字体（由小圆点/灯珠组成的数字，模拟LED显示屏）
#
# 字体文件获取建议：
# - 正常字体：Windows系统自带或从网上下载常见字体
# - 数字字体：可在 dafont.com, fontsquirrel.com 等网站搜索 "digital", "LCD", "LED" 等关键词
# - 点阵字体：搜索 "dot matrix font", "pixel font" 等
# - LED灯珠字体：搜索 "LED dot font", "dot LED display font" 等

# 正常数字字体（系统常见字体）
FONT_NORMAL = [
    "arial.ttf",           # Arial
    "arialbd.ttf",         # Arial Bold
    "times.ttf",           # Times New Roman
    "timesbd.ttf",         # Times New Roman Bold
    "verdana.ttf",         # Verdana
    "verdanab.ttf",        # Verdana Bold
    "calibri.ttf",         # Calibri
    "calibrib.ttf",        # Calibri Bold
    "fonts/TimesNewRoman.ttf",
    "fonts/Verdana.ttf"
]

# 数字显示字体（7段数码管风格）
FONT_DIGITAL = [
    "fonts/digital-7-mono-3.ttf", 
    "fonts/DS-DIGI.TTF",
    "fonts/DS-DIGIB.TTF",
    "fonts/LED.ttf",
]

# 点阵字体（Dot Matrix Fonts）
FONT_DOT_MATRIX = [
    "fonts/DotMatrix.ttf",
    "fonts/lcddot_tr.ttf",
    "fonts/dotty.ttf",
    "fonts/Pixel-lcd-machine.ttf",
]

# LED 灯珠字体（由小圆点组成的数字）
FONT_LED_DOT = [
    "fonts/led-dot.ttf",
    "fonts/LED-Dot-Matrix.ttf",
    "fonts/The-Led-Display-St.ttf",
    "fonts/led-dot-display.ttf",
]

# 所有字体类型（用于随机选择）
ALL_FONT_TYPES = [
    ("normal", FONT_NORMAL),
    ("digital", FONT_DIGITAL),
    ("dot_matrix", FONT_DOT_MATRIX),
    ("led_dot", FONT_LED_DOT)
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

def get_score_labels():
    # 000-999 (3位数字比分，覆盖所有可能)
    return [f"{i:03d}" for i in range(1000)]

def get_shotclock_labels():
    # 00-24
    return [f"{i:02d}" for i in range(25)]

def get_time_labels():
    """生成时间标签，仅支持 MM:SS 格式（不包含毫秒）"""
    labels = set()

    # MM:SS 格式（分秒）
    for m in range(13):  # 0-12分钟
        for s in range(60):  # 0-59秒，覆盖所有时间
            labels.add(f"{m:02d}:{s:02d}")
            # 额外加入单数字分钟格式（M:SS），满足「分钟只有一个数字」的需求
            if m < 10:
                labels.add(f"{m}:{s:02d}")

    return list(labels)

# ================= 图像处理核心 =================

def load_font(text, size, font_type=None):
    """
    加载字体（支持多种字体类型）
    
    Args:
        text: 要渲染的文本
        size: 字体大小
        font_type: 字体类型 ('normal', 'digital', 'dot_matrix', 'led_dot')，如果为None则随机选择
    """
    # 如果未指定字体类型，随机选择一种
    if font_type is None:
        font_type = random.choice(["normal", "digital", "dot_matrix", "led_dot"])
    
    # 根据字体类型选择字体列表
    font_map = {
        "normal": FONT_NORMAL,
        "digital": FONT_DIGITAL,
        "dot_matrix": FONT_DOT_MATRIX,
        "led_dot": FONT_LED_DOT
    }
    
    font_list = font_map.get(font_type, FONT_NORMAL)
    
    # 尝试加载指定类型的字体
    for f_path in font_list:
        try:
            if os.path.exists(f_path):
                return ImageFont.truetype(f_path, size)
        except:
            continue
    
    # 如果指定类型都失败，尝试其他类型
    for font_type_name, fonts in ALL_FONT_TYPES:
        if font_type_name == font_type:
            continue
        for f_path in fonts:
            try:
                if os.path.exists(f_path):
                    return ImageFont.truetype(f_path, size)
            except:
                continue
    
    # 如果都失败，尝试系统字体
    try:
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

    # 7. 亮度调节（模拟不同光照条件）
    if random.random() > 0.2:  # 80%概率应用亮度调节
        brightness_delta = random.uniform(-40, 40)  # 亮度调整范围：-40到+40
        cv_img = cv2.convertScaleAbs(cv_img, alpha=1.0, beta=brightness_delta)

    # 8. 对比度调节（模拟不同显示效果）
    if random.random() > 0.2:  # 80%概率应用对比度调节
        contrast_alpha = random.uniform(0.7, 1.3)  # 对比度系数：0.7到1.3
        cv_img = cv2.convertScaleAbs(cv_img, alpha=contrast_alpha, beta=0)

    return cv_img

def save_image_unicode(img_cv, save_path, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img_cv)
    if not ok:
        raise ValueError(f"Failed to encode image for {save_path}")
    buf.tofile(save_path)


def generate_image(text, save_path):
    # 原项目定义的图像尺寸：宽度94，高度24
    TARGET_WIDTH = 94
    TARGET_HEIGHT = 24
    
    # 使用较大的临时画布来绘制文本，然后缩放
    temp_h = 64
    font_size = random.randint(40, 50)
    kerning = random.randint(-5, 2) # 粘连控制
    
    # 随机选择字体类型（正常、数字、点阵、LED灯珠）
    font_type = random.choice(["normal", "digital", "dot_matrix", "led_dot"])
    
    # 主体字体与毫秒字体（毫秒更小，使用相同字体类型保持一致性）
    font_main = load_font(text, font_size, font_type=font_type)
    ms_font_size = max(20, int(font_size * random.uniform(0.55, 0.8)))
    font_ms = load_font(text, ms_font_size, font_type=font_type)

    temp_w = int(font_size * 0.85 * len(text)) + 60
    img_pil = Image.new('RGB', (temp_w, temp_h), (0, 0, 0))
    draw = ImageDraw.Draw(img_pil)
    color = random.choice(COLORS)
    
    current_x = 10
    y_main = (temp_h - font_size) // 2 - 5
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
    
    # Crop 到文本边界
    bbox = img_pil.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        img_pil = img_pil.crop((left-5, 0, right+5, temp_h))
    
    # 转换为OpenCV格式并应用增强
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_cv = apply_augmentations(img_cv)
    
    # 统一调整到目标尺寸 94x24（直接拉伸，保持训练和推理一致）
    img_cv = cv2.resize(img_cv, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
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
    parser.add_argument("--train_samples", type=int, default=SAMPLES_PER_CLASS_TRAIN,
                        help="每类训练样本数")
    parser.add_argument("--test_samples", type=int, default=SAMPLES_PER_CLASS_TEST,
                        help="每类测试样本数")
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT,
                        help="输出根目录")
    parser.add_argument("--clean", action="store_true",
                        help="生成前清空输出目录")
    parser.add_argument("--append", action="store_true",
                        help="追加生成：保留已存在的图片，若数量不足则补齐")
    return parser.parse_args()

# ================= 主程序 =================

def main():
    args = parse_args()

    # 准备任务（仅数字、冒号和点）
    score_labels = get_score_labels()
    shotclock_labels = get_shotclock_labels()
    time_labels = get_time_labels()
    # 移除节次标签（包含英文字母）
    tasks = [score_labels, shotclock_labels, time_labels]

    all_labels = set()
    for t in tasks:
        all_labels.update(t)
    
    # 打印统计信息
    print("=" * 60)
    print("数据集生成统计（仅数字、冒号）:")
    print(f"  比分标签: {len(score_labels)} 个 (000-999)")
    print(f"  24秒倒计时标签: {len(shotclock_labels)} 个 (00-24)")
    print(f"  时间标签: {len(time_labels)} 个 (MM:SS + M:SS)")
    print(f"  总唯一标签数: {len(all_labels)} 个")
    print("=" * 60)
    
    # 可选清理输出目录
    if args.clean and os.path.exists(args.output_root):
        shutil.rmtree(args.output_root, ignore_errors=True)

    # 生成训练集，直接写入对应标签文件夹，文件名用数字
    print("Generating train set...")
    total_labels = len(all_labels)
    generated_count = 0
    error_count = 0
    
    for idx, label_text in enumerate(all_labels, 1):
        safe_folder_name = sanitize_filename(label_text)
        folder_path = os.path.join(args.output_root, 'train', safe_folder_name)
        os.makedirs(folder_path, exist_ok=True)
        try:
            # 追加模式：如果已有样本则保留，只补足缺的数量
            existing_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            existing_count = len(existing_files)
            start_idx = existing_count if args.append else 0
            need = max(args.train_samples - existing_count, 0) if args.append else args.train_samples

            for i in range(need):
                file_name = f"{start_idx + i:03d}.jpg"
                full_path = os.path.join(folder_path, file_name)
                generate_image(label_text, full_path)
            generated_count += 1
        except Exception as e:
            print(f"Error generating images for {label_text}: {e}")
            error_count += 1
            continue
            
        # 每50个标签打印一次进度，更频繁的反馈
        if idx % 50 == 0:
            print(f"Progress: {idx}/{total_labels} labels processed (generated: {generated_count}, errors: {error_count})...")
    
    print(f"Finished train. Processed {total_labels} labels (successfully generated: {generated_count}, errors: {error_count}).")

    # 生成测试集：从训练集中拷贝部分样本
    print("Generating test set (copy from train)...")
    total_labels = len(all_labels)
    copied_count = 0
    error_count = 0
    
    for idx, label_text in enumerate(all_labels, 1):
        safe_folder_name = sanitize_filename(label_text)
        src_folder = os.path.join(args.output_root, 'train', safe_folder_name)
        dst_folder = os.path.join(args.output_root, 'test', safe_folder_name)
        
        if not os.path.exists(src_folder):
            print(f"Warning: Source folder {src_folder} does not exist, skipping...")
            error_count += 1
            continue
            
        os.makedirs(dst_folder, exist_ok=True)
        try:
            files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not files:
                print(f"Warning: No image files found in {src_folder}, skipping...")
                error_count += 1
                continue
                
            take_n = min(args.test_samples, len(files))
            for f in files[:take_n]:
                src_path = os.path.join(src_folder, f)
                dst_path = os.path.join(dst_folder, f)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying files for {label_text}: {e}")
            error_count += 1
            continue
            
        # 每50个标签打印一次进度，更频繁的反馈
        if idx % 50 == 0:
            print(f"Progress: {idx}/{total_labels} labels processed (copied: {copied_count}, errors: {error_count})...")
    
    print(f"Finished test. Processed {total_labels} labels (successfully copied: {copied_count}, errors: {error_count}).")

if __name__ == "__main__":
    main()