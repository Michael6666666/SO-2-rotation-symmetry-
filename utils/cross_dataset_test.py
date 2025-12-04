import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from datasets.unified_rotated_dataset import UnifiedRotatedDataset

def evaluate_on_dataset(model, test_loader, device, dataset_name="",label_offset=0):
    model.eval()
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f'Evaluating on {dataset_name}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            labels=(labels+label_offset).to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for label, pred in zip(labels, predicted):
                label = label.item()
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0

                class_total[label] += 1
                if pred.item() == label:
                    class_correct[label] += 1

    accuracy = 100. * correct / total

    per_class_acc = {}
    for label in class_total:
        per_class_acc[label] = 100. * class_correct[label] / class_total[label]

    return accuracy, per_class_acc


def cross_dataset_evaluation(model, config):
    print("="*70)
    print("🔬 Begin cross data set evaluating")
    print("="*70)

    device = config.device
    model = model.to(device)
    dataset_configs = [('dtd',0),('kth',47),('cifar10',57)]
    results = {}

    
    for dataset_name, offset in dataset_configs:
        print(f"\n📊 Loading {dataset_name.upper()} testing set (Offset: {offset})...")

        test_dataset = UnifiedRotatedDataset(
            dataset_name=dataset_name,
            split='test',
            img_size=config.img_size,
            rotation_range=config.rotation_range,
            seed=config.seed,
            data_root=config.data_root
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if device == 'cuda' else False
        )

        accuracy, per_class_acc = evaluate_on_dataset(
            model, test_loader, device, dataset_name.upper(),label_offset=offset
        )

        results[dataset_name] = {
            'accuracy': accuracy,
            'per_class_acc': per_class_acc,
            'num_classes': test_dataset.get_num_classes(),
            'num_samples': len(test_dataset)
        }

        print(f"✅ {dataset_name.upper()}: {accuracy:.2f}%")

    return results


def print_cross_dataset_results(results, trained_on="Mixed"):
    
    print("\n" + "="*70)
    print(f"📊 cross dataset evaluation result, trained on: {trained_on})")
    print("="*70)

    print(f"\n{'Dataset':<15} {'Accuracy':<12} {'Samples':<10} {'Classes':<10}")
    print("-"*70)

    for dataset_name, result in results.items():
        print(f"{dataset_name.upper():<15} "
              f"{result['accuracy']:<12.2f}% "
              f"{result['num_samples']:<10} "
              f"{result['num_classes']:<10}")

    print("="*70)

    
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"\n📈 Average accuracy: {avg_acc:.2f}%")

    
    best_dataset = max(results.items(), key=lambda x: x[1]['accuracy'])
    worst_dataset = min(results.items(), key=lambda x: x[1]['accuracy'])

    print(f"🏆 Best: {best_dataset[0].upper()} ({best_dataset[1]['accuracy']:.2f}%)")
    print(f"⚠️  Worst: {worst_dataset[0].upper()} ({worst_dataset[1]['accuracy']:.2f}%)")


def compare_models_cross_dataset(baseline_model, steerable_model, config):
    
    print("="*70)
    print("⚔️  Model comparsion: Baseline vs Steerable")
    print("="*70)

    print("\n1️⃣  Evaluate Baseline model...")
    baseline_results = cross_dataset_evaluation(baseline_model, config)

    print("\n2️⃣  Evaluate Steerable model...")
    steerable_results = cross_dataset_evaluation(steerable_model, config)

    print("\n" + "="*70)
    print("📊 Model performance comparsion")
    print("="*70)

    print(f"\n{'Dataset':<15} {'Baseline':<15} {'Steerable':<15} {'Improvement':<15}")
    print("-"*70)

    comparison_results = {}

    for dataset_name in baseline_results.keys():
        baseline_acc = baseline_results[dataset_name]['accuracy']
        steerable_acc = steerable_results[dataset_name]['accuracy']
        improvement = steerable_acc - baseline_acc

        print(f"{dataset_name.upper():<15} "
              f"{baseline_acc:<15.2f}% "
              f"{steerable_acc:<15.2f}% "
              f"{improvement:+.2f}%")

        comparison_results[dataset_name] = {
            'baseline': baseline_acc,
            'steerable': steerable_acc,
            'improvement': improvement
        }
    
    avg_baseline = np.mean([r['accuracy'] for r in baseline_results.values()])
    avg_steerable = np.mean([r['accuracy'] for r in steerable_results.values()])
    avg_improvement = avg_steerable - avg_baseline

    print("-"*70)
    print(f"{'AVERAGE':<15} "
          f"{avg_baseline:<15.2f}% "
          f"{avg_steerable:<15.2f}% "
          f"{avg_improvement:+.2f}%")
    print("="*70)

    if avg_improvement > 0:
        print(f"\n Steerable CNN average improvement {avg_improvement:.2f}%")
    else:
        print(f"\n Baseline has better performance {abs(avg_improvement):.2f}%")

    return {
        'baseline': baseline_results,
        'steerable': steerable_results,
        'comparison': comparison_results,
        'avg_improvement': avg_improvement
    }

def cross_dataset_evaluation_single_trained(model, train_dataset_name, config):
    """
    For models trained on single dataset (e.g., DTD=47 classes)
    Test on all 3 datasets
    """
    print("="*70)
    print(f"🔬 Cross-dataset evaluation (Trained on {train_dataset_name.upper()})")
    print("="*70)

    device = config.device
    model = model.to(device)
    model.eval()
    
    test_datasets = ['dtd', 'kth', 'cifar10']
    results = {}
    
    for test_name in test_datasets:
        print(f"\n📊 Testing on {test_name.upper()}...")
        
        test_set = UnifiedRotatedDataset(test_name, 'test', config.img_size, config.rotation_range, data_root=config.data_root)
        test_loader = DataLoader(test_set, config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True if device=='cuda' else False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'{test_name.upper()}', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # 如果测试集类别数 <= 模型类别数，直接预测
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        results[test_name] = {'accuracy': acc, 'num_samples': len(test_set)}
        print(f"✅ {test_name.upper()}: {acc:.2f}%")
    
    return results