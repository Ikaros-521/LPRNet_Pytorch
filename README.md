# Basketball Scoreboard Recognition

基于LPRNet架构的篮球计分板识别系统（Basketball Scoreboard Recognition System）

本项目基于LPRNet架构改造，用于识别篮球比赛计分板上的各种信息（仅支持英文和数字）：
- **比分**：3位数字（如 123）
- **24秒倒计时**：2位数字（如 24）
- **倒计时**：分秒、分秒+毫秒（1-3位，如 34:56.7 / 34:56.78 / 34:56.789）
- **节次**：仅支持英文格式
  - 英文：FIRST、SECOND、THIRD、FOURTH、OT、OVERTIME
  - 加时：OT1-OT7、OVERTIME1-OVERTIME7
  - 缩写：1st、2nd、3rd、4th、overtime、ot1-ot7、overtime1-overtime7

# dependencies

- pytorch >= 1.0.0
- opencv-python 3.x
- python 3.x
- imutils
- Pillow
- numpy

# pretrained model

预训练模型保存在 `weights/` 目录下。如果没有预训练模型，需要先进行训练。

# training and testing

## 数据准备

1. 准备训练和测试数据集
2. 图片尺寸建议：94x24（可根据实际计分板尺寸调整）
3. **数据组织方式**：使用文件夹名作为标签，同一标签的多个图片放在同一文件夹中

### 目录结构示例

```
data/
├── train/
│   ├── 123/              # 文件夹名是标签"123"（比分）
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── 24/               # 标签"24"（24秒倒计时）
│   │   └── ...
│   ├── 12:34:56/         # 标签"12:34:56"（时间）
│   │   └── ...
│   ├── FIRST/            # 标签"FIRST"（节次，仅英文）
│   │   └── ...
│   └── ...
└── test/
    └── ...
```

图片文件名可以是任意名称，系统会自动从文件夹名中提取标签。

## 训练

```bash
python train_LPRNet.py --train_img_dirs ./data/train --test_img_dirs ./data/test --max_len 12
```

主要参数：
- `--train_img_dirs`: 训练图片目录
- `--test_img_dirs`: 测试图片目录
- `--max_len`: 最大文本长度（默认12，可覆盖比分、时间、节次等）
- `--img_size`: 图片尺寸（默认 [94, 24]）
- `--max_epoch`: 训练轮数（默认15）
- `--learning_rate`: 学习率（默认0.1）

## 测试

```bash
python test_LPRNet.py --test_img_dirs ./data/test --pretrained_model ./weights/Final_Scoreboard_model.pth
```

显示测试结果：
```bash
python test_LPRNet.py --test_img_dirs ./data/test --show true
```

# 字符集

支持的字符（仅英文和数字）：
- 数字：0-9
- 时间分隔符：`:` (冒号)、`.` (点)
- 大写字母：A-Z（用于FIRST、SECOND、OT、OVERTIME等）
- 小写字母：a-z（用于1st、2nd、3rd、4th、overtime、ot1等）

# 项目结构

```
.
├── data/              # 数据加载相关
│   ├── load_data.py  # 数据加载器（已改造为ScoreboardDataLoader）
│   └── test/         # 测试图片目录
├── model/            # 模型定义
│   └── LPRNet.py     # LPRNet模型架构
├── weights/          # 模型权重保存目录
├── train_LPRNet.py   # 训练脚本
├── test_LPRNet.py    # 测试脚本
└── README.md         # 项目说明
```

# References

1. [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
2. [PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)

# 注意事项

- 本项目基于LPRNet架构改造，保持了原有的轻量级和高性能特点
- **仅支持英文和数字识别**，不支持中文节次
- 字符集包含：数字0-9、冒号(:)、点(.)、英文字母（大小写）
- 数据加载器使用直接拉伸到固定尺寸（94x24），保持训练和推理一致
- 建议根据实际计分板样式调整图片尺寸和最大长度参数
