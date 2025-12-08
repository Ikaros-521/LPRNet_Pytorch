from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os
import re

# 篮球计分板字符集
# 包含：数字0-9、冒号(:)、点(.)、中文字符（第、一、二、三、四、节）、英文字母（FIRST、1st等）
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         ':', '.',  # 时间分隔符
         '第', '一', '二', '三', '四', '节', '五', '六', '七', '八', '九', '十', '加', '时', '赛',  # 中文节次
         'F', 'I', 'R', 'S', 'T',  # FIRST
         'A', 'B', 'C', 'D', 'E', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'U', 'V',
         'W', 'X', 'Y', 'Z',
         's', 't', 'n', 'd', 'r', 'h', 'o',  # 小写字母用于1st, 2nd, 3rd, 4th，ot1等
         'e', 'v', 'i', 'm', # overtime
         '-'  # 空白/分隔符（CTC需要）
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

def convert_folder_name_to_label(folder_name):
    """
    将文件夹名转换为标签
    主要处理Windows文件夹命名限制：将下划线(_)转换回冒号(:)
    
    规则：
    - 如果下划线前后都是数字，则转换为冒号（用于时间格式，如 34_56 -> 34:56）
    - 其他情况保持原样
    
    示例：
    - '34_56' -> '34:56'
    - '34_56.789' -> '34:56.789'
    - '12_34_56' -> '12:34:56' (多次转换)
    - 'FIRST' -> 'FIRST' (不变)
    """
    # 匹配模式：数字_数字（时间格式）
    # 例如：34_56 -> 34:56, 34_56.789 -> 34:56.789, 12_34_56 -> 12:34:56
    pattern = r'(\d+)_(\d+)'
    label_str = re.sub(pattern, r'\1:\2', folder_name)
    return label_str

def imread_unicode(path):
    """Robust cv2 imread for unicode paths on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


class ScoreboardDataLoader(Dataset):
    """
    篮球计分板数据加载器
    支持识别：比分（3位数字）、24秒倒计时（2位数字）、倒计时（时分秒、时分秒毫秒）、节次等
    
    数据组织方式：
    使用文件夹名作为标签，同一标签的多个图片放在同一文件夹中
    例如：
        data/train/
            ├── 123/          # 文件夹名是标签
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            ├── 12:34:56/
            │   └── ...
            └── 第一节/
                └── ...
    """
    def __init__(self, img_dir, imgSize, max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        self.labels = []  # 存储每个图片对应的标签
        
        # 遍历每个目录，收集图片路径和对应的标签（文件夹名）
        for dir_path in img_dir:
            dir_path = os.path.expanduser(dir_path)
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} does not exist, skipping...")
                continue
            
            # 遍历目录下的所有子文件夹
            for folder_name in os.listdir(dir_path):
                folder_path = os.path.join(dir_path, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                
                # 文件夹名转换为标签（处理Windows命名限制：下划线转冒号）
                label_str = convert_folder_name_to_label(folder_name)
                
                # 验证标签是否包含有效字符
                if not any(c in CHARS_DICT for c in label_str):
                    print(f"Warning: Folder '{folder_name}' (converted to '{label_str}') contains no valid characters, skipping...")
                    continue
                
                # 将标签字符串转换为索引列表
                label = []
                invalid_chars = []
                for c in label_str:
                    if c in CHARS_DICT:
                        label.append(CHARS_DICT[c])
                    else:
                        invalid_chars.append(c)
                
                # 如果有无效字符，打印警告（只打印一次）
                if invalid_chars:
                    print(f"Warning: Folder '{folder_name}' contains invalid characters: {set(invalid_chars)}, these will be skipped")
                
                if len(label) == 0:
                    print(f"Warning: Folder '{folder_name}' resulted in empty label after filtering, skipping...")
                    continue
                
                # 收集该文件夹下的所有图片
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(file_name)
                        if ext.lower() in image_extensions:
                            self.img_paths.append(file_path)
                            self.labels.append(label)
        
        # 打乱数据顺序
        combined = list(zip(self.img_paths, self.labels))
        random.shuffle(combined)
        self.img_paths, self.labels = zip(*combined) if combined else ([], [])
        self.img_paths = list(self.img_paths)
        self.labels = list(self.labels)
        
        self.img_size = imgSize
        self.max_len = max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform
        
        print(f"Loaded {len(self.img_paths)} images from {len(set([tuple(l) for l in self.labels]))} unique labels")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        label = self.labels[index]
        
        Image = imread_unicode(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        return Image, label, len(label)

    def transform(self, img):
        # 直接拉伸到固定尺寸（与训练/推理保持一致）
        img = cv2.resize(img, tuple(self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        """
        验证标签格式（可选，用于数据质量检查）
        篮球计分板格式较灵活，这里提供基本验证
        """
        if len(label) == 0:
            print("Error: Empty label!")
            return False
        return True

# 保持向后兼容的别名
LPRDataLoader = ScoreboardDataLoader
