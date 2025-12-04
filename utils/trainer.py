# train_utils.py
"""Training and evaluation function"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import time
import copy

class Trainer:
    

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # Lose function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

        # Recording history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        # Best model check point
        self.best_acc = 0.0
        self.best_model_weights = None

        print(f"✅ Trainer initialized finished")
        print(f"   Optimizer: Adam (lr={config.learning_rate})")
        print(f"   Lose function: CrossEntropyLoss")

    def train_epoch(self):
        """Training one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward propagation
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward propagation
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validation"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, num_epochs=None):
        """Full training pipeline"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print(f"\n🚀 Begin training - {num_epochs} epochs")
        print("="*60)

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_loss, val_acc = self.validate()

            # Learning rate change
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            
            if current_lr != old_lr:
                print(f"📉 Learning rate decrease: {old_lr:.6f} → {current_lr:.6f}")

            # Record training history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Print out the result
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")

            # Saved the best accuracy model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                print(f"✅ Best model: {val_acc:.2f}%")

        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ Training finished！")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Best validation accuracy: {self.best_acc:.2f}%")

        
        self.model.load_state_dict(self.best_model_weights)

        return self.history

    def save_checkpoint(self, filename):
        """Keep checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, filename)
        print(f"💾 Model saved at: {filename}")