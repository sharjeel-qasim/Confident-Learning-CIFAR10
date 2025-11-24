import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

BASE_DIR = 'cl_results'
INPUT_DATA_DIR = os.path.join(BASE_DIR, 'prepared_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'final_models')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True) 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100 # Final training duration
NUM_CLASSES = 10

# 9 MODELS TO TRAIN ---
MODELS_TO_RUN = [
    # 0% Noise
    {'name': 'M1_baseline_0p', 'type': 'baseline', 'noise': 0},
    {'name': 'M2_pruned_0p',   'type': 'pruned',   'noise': 0},
    {'name': 'M3_corrected_0p','type': 'corrected','noise': 0},
    # 20% Noise
    {'name': 'M4_baseline_20p', 'type': 'baseline', 'noise': 20},
    {'name': 'M5_pruned_20p',   'type': 'pruned',   'noise': 20},
    {'name': 'M6_corrected_20p','type': 'corrected','noise': 20},
    # 40% Noise
    {'name': 'M7_baseline_40p', 'type': 'baseline', 'noise': 40},
    {'name': 'M8_pruned_40p',   'type': 'pruned',   'noise': 40},
    {'name': 'M9_corrected_40p','type': 'corrected','noise': 40},
]

def create_model():
    """Initializes a ResNet-18 model configured for CIFAR-10."""
    net = timm.create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net.to(DEVICE)

def load_prepared_data(data_type, noise_level):
    """Loads the appropriate labels and indices based on run type."""
    noise_prefix = f"{noise_level}p"
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    
    if data_type == 'baseline':
        label_file = f'noisy_labels_{noise_prefix}.npy'
        indices = np.arange(len(full_trainset))
    elif data_type == 'pruned':
        label_file = f'noisy_labels_{noise_prefix}.npy'
        indices = np.load(os.path.join(INPUT_DATA_DIR, f'pruned_indices_{noise_prefix}.npy'))
    elif data_type == 'corrected':
        label_file = f'corrected_labels_{noise_prefix}.npy'
        indices = np.arange(len(full_trainset))
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    all_labels = np.load(os.path.join(INPUT_DATA_DIR, label_file))
    full_trainset.targets = all_labels.tolist()
    
    return torch.utils.data.Subset(full_trainset, indices)

def plot_learning_curve(model_name, epochs, test_accs):
    """Plots Test Accuracy vs. Epochs and saves the chart."""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, test_accs, label='Test Accuracy', marker='o', linestyle='-', linewidth=2)
    
    best_acc = max(test_accs)
    best_epoch = epochs[test_accs.index(best_acc)]
    plt.scatter(best_epoch, best_acc, color='red', s=100, label=f'Best Acc: {best_acc:.2f}%')
    
    plt.title(f'Learning Curve: {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, alpha=0.5)
    plt.legend()
    
    plot_path = os.path.join(RESULTS_DIR, f"{model_name}_learning_curve.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Learning curve saved: {plot_path}")


def train_and_eval(model_config):
    
    model_name = model_config['name']
    data_type = model_config['type']
    noise_level = model_config['noise']

    log_path = os.path.join(RESULTS_DIR, f"{model_name}_log.txt")
    final_output_path = os.path.join(RESULTS_DIR, f"{model_name}_FINAL_ACCURACY.txt")

    if os.path.exists(final_output_path):
        print(f"Skipping {model_name}: Already completed.")
        return

    def log(msg):
        print(msg)
        with open(log_path, "a") as f: f.write(msg + "\n")

    log(f"--- Starting Training: {model_name} (Type: {data_type}, Noise: {noise_level}%) ---")
    
    train_subset = load_prepared_data(data_type, noise_level)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    log(f"Training on {len(train_subset)} examples.")

    net = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    best_acc = 0.0
    best_model_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pth")
    
    epoch_history = []
    test_acc_history = []

    for epoch in tqdm(range(NUM_EPOCHS), desc=f"{model_name} Training"):
        net.train()
        
        for inputs, targets in trainloader:
             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
             optimizer.zero_grad()
             outputs = net(inputs)
             loss = criterion(outputs, targets)
             loss.backward()
             optimizer.step()
        
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
            net.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = net(inputs); pred = outputs.argmax(dim=1)
                    correct += pred.eq(targets).sum().item(); total += targets.size(0)

            acc = 100.0 * correct / total
            log(f"Epoch {epoch+1:3d} -> Test Accuracy: {acc:.3f}%")
            
            epoch_history.append(epoch + 1)
            test_acc_history.append(acc)

            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), best_model_path)
                log(f"NEW BEST -> {acc:.3f}% - Model saved!")

    # Final actions
    plot_learning_curve(model_name, epoch_history, test_acc_history)

    log("\n" + "="*60)
    log(f"FINAL RESULT: {model_name} Best Test Accuracy: {best_acc:.3f}%")
    log("="*60)
    
    with open(final_output_path, 'w') as f:
        f.write(f"{best_acc:.3f}")


if __name__ == '__main__':
    
    all_results = {}
    print("\n" + "="*80)
    print("STARTING SEQUENTIAL CONFIDENT LEARNING EVALUATION (9 MODELS)")
    print(f"Total Epochs per Model: {NUM_EPOCHS} | Device: {DEVICE}")
    print("="*80)
    
    for config in MODELS_TO_RUN:
        try:
            train_and_eval(config)
            
        except FileNotFoundError as e:
            print(f"\n--- FATAL ERROR: Data files missing for {config['name']} ---")
            print("Please ensure Step 1 and Step 2 completed successfully and check the path.")
            print(e)
            break
        except Exception as e:
            print(f"\n--- CRITICAL ERROR during training model {config['name']} ---")
            print(f"Error: {e}")
            print("Skipping to next model (check logs for details)...")
            continue
    
    print("\n" + "="*60)
    print("ALL SEQUENTIAL TRAINING JOBS FINISHED.")
    print("="*60)