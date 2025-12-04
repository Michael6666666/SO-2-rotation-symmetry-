import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def compare_two_models(history1, history2, name1='Model 1', name2='Model 2',
                       test_acc1=None, test_acc2=None, save_name='comparison.png'):
    """Compare two models' training curves"""
    
    save_path = Path('./results') / save_name
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{name1} vs {name2}', fontsize=16, fontweight='bold')
    
    epochs1 = range(1, len(history1['train_loss']) + 1)
    epochs2 = range(1, len(history2['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs1, history1['train_loss'], 'b-', label=f'{name1} Train', linewidth=2, alpha=0.7)
    axes[0].plot(epochs1, history1['val_loss'], 'b--', label=f'{name1} Val', linewidth=2, alpha=0.7)
    axes[0].plot(epochs2, history2['train_loss'], 'r-', label=f'{name2} Train', linewidth=2, alpha=0.7)
    axes[0].plot(epochs2, history2['val_loss'], 'r--', label=f'{name2} Val', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs1, history1['train_acc'], 'b-', label=f'{name1} Train', linewidth=2, alpha=0.7)
    axes[1].plot(epochs1, history1['val_acc'], 'b--', label=f'{name1} Val', linewidth=2, alpha=0.7)
    axes[1].plot(epochs2, history2['train_acc'], 'r-', label=f'{name2} Train', linewidth=2, alpha=0.7)
    axes[1].plot(epochs2, history2['val_acc'], 'r--', label=f'{name2} Val', linewidth=2, alpha=0.7)
    
    best1 = max(history1['val_acc'])
    best2 = max(history2['val_acc'])
    axes[1].plot(np.argmax(history1['val_acc']) + 1, best1, 'b*', markersize=12)
    axes[1].plot(np.argmax(history2['val_acc']) + 1, best2, 'r*', markersize=12)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs1, history1['lr'], 'b-', label=name1, linewidth=2, alpha=0.7)
    axes[2].plot(epochs2, history2['lr'], 'r-', label=name2, linewidth=2, alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    
    # Print stats
    print("\n" + "="*80)
    print(f"📊 {name1} vs {name2}")
    print("="*80)
    print(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
    print("-"*80)
    print(f"{'Final Train Accuracy':<30} {history1['train_acc'][-1]:<20.2f}% {history2['train_acc'][-1]:<20.2f}% {history2['train_acc'][-1]-history1['train_acc'][-1]:+.2f}%")
    print(f"{'Final Val Accuracy':<30} {history1['val_acc'][-1]:<20.2f}% {history2['val_acc'][-1]:<20.2f}% {history2['val_acc'][-1]-history1['val_acc'][-1]:+.2f}%")
    print(f"{'Best Val Accuracy':<30} {best1:<20.2f}% {best2:<20.2f}% {best2-best1:+.2f}%")
    if test_acc1 is not None and test_acc2 is not None:
        print(f"{'Test Accuracy':<30} {test_acc1:<20.2f}% {test_acc2:<20.2f}% {test_acc2-test_acc1:+.2f}%")
    print("="*80)
