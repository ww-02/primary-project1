import os
import torch
import torch.nn as nn
import time


class ClassificationTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, cfg):
        """
        分类任务训练器（纯PyTorch版本）
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

        # 将模型移动到指定设备
        self.model = self.model.to(device)

        self.ckpt_dir = cfg["save"]["ckpt_dir"]
        self.epochs = cfg["train"]["epochs"]

        # 训练历史记录
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 打印进度
            if (batch_idx + 1) % 5 == 0:
                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """验证函数"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
        }

        # 保存最新模型
        latest_path = os.path.join(self.ckpt_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.ckpt_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"  → 保存最佳模型到: {best_path}")

    def train(self):
        """完整的训练流程"""
        print("=" * 60)
        print("开始训练...")
        print(f"总轮数: {self.epochs}")
        print(f"设备: {self.device}")
        print(f"训练集: {len(self.train_loader.dataset)} 张图片")
        print(f"验证集: {len(self.val_loader.dataset)} 张图片")
        print("=" * 60)

        best_val_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            print("-" * 40)

            # 训练
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            train_time = time.time() - start_time

            # 验证
            val_loss, val_acc = self.validate()

            # 保存历史
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # 打印结果
            print(f"训练时间: {train_time:.1f}秒")
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  → 新的最佳准确率: {best_val_acc:.2f}%")
            elif epoch % 5 == 0:  # 每5轮保存一次检查点
                self.save_checkpoint(epoch, is_best=False)

        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"模型保存位置: {self.ckpt_dir}")
        print("=" * 60)