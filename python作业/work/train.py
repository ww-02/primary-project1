import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.image_dataset import ImageClsDataset
from datasets.transform import build_transforms
from models.models import SimpleCNN
from losses.losses import ClassificationLoss
from trainers.cls_trainers import ClassificationTrainer


def main():
    # 直接在代码中定义配置
    cfg = {
        "train": {
            "epochs": 10,
            "batch_size": 8,
            "lr": 0.001,
            "num_workers": 0,  # Windows必须设0
            "device": "cpu"
        },
        "data": {
            "root": "E:/python作业/data",  # 请根据实际路径修改
            "num_classes": 2,
            "img_size": 224
        },
        "save": {
            "log_dir": "E:/python作业/work_dirs/logs",
            "ckpt_dir": "E:/python作业/work_dirs/ckpt"
        }
    }

    # 创建保存目录
    os.makedirs(cfg["save"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["save"]["ckpt_dir"], exist_ok=True)

    # 设备
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据
    train_transform = build_transforms(cfg["data"]["img_size"], is_train=True)
    val_transform = build_transforms(cfg["data"]["img_size"], is_train=False)

    # 数据集
    train_dataset = ImageClsDataset(
        root=cfg["data"]["root"],
        split="train",
        transform=train_transform
    )
    val_dataset = ImageClsDataset(
        root=cfg["data"]["root"],
        split="val",
        transform=val_transform
    )

    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"]
    )

    # 模型
    model = SimpleCNN(num_classes=cfg["data"]["num_classes"])

    # 损失函数和优化器
    loss_fn = ClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    # 训练 - 这里要注意参数顺序
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        device=device,
        cfg=cfg
    )
    trainer.train()


if __name__ == "__main__":
    main()