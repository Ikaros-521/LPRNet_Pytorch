"""快速检查测试集生成状态"""
import os

def check_test_generation():
    output_root = "data_win"
    train_dir = os.path.join(output_root, "train")
    test_dir = os.path.join(output_root, "test")
    
    if not os.path.exists(train_dir):
        print(f"训练集目录不存在: {train_dir}")
        return
    
    # 统计训练集标签
    train_labels = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"训练集标签数: {len(train_labels)}")
    
    # 统计测试集标签
    if os.path.exists(test_dir):
        test_labels = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        print(f"测试集标签数: {len(test_labels)}")
        
        # 检查缺失的标签
        missing = set(train_labels) - set(test_labels)
        if missing:
            print(f"缺失的测试集标签数: {len(missing)}")
            print(f"前10个缺失标签: {list(missing)[:10]}")
        else:
            print("所有标签都已生成测试集")
            
        # 检查测试集文件数
        total_test_files = 0
        empty_folders = []
        for label in test_labels[:100]:  # 只检查前100个
            label_path = os.path.join(test_dir, label)
            files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_test_files += len(files)
            if len(files) == 0:
                empty_folders.append(label)
        
        print(f"前100个标签的测试文件总数: {total_test_files}")
        if empty_folders:
            print(f"发现空文件夹: {empty_folders[:5]}")
    else:
        print(f"测试集目录不存在: {test_dir}")

if __name__ == "__main__":
    check_test_generation()

