# test_ev_no_timm.py   ← works without timm
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

MODEL_PATH = "./cl_results/final_models/M6_corrected_20p_best.pth"  
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if __name__ == '__main__':

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    model = models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model.to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded: {MODEL_PATH}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in tqdm(testloader, desc="Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTEST ACCURACY: {accuracy*100:.3f}%\n")

    print("Per-class metrics:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'shrink': 0.8})
    plt.title(f'CIFAR-10 Confusion Matrix\n{os.path.basename(MODEL_PATH)}\nAccuracy = {accuracy*100:.2f}%',
              fontsize=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    save_dir = './cl_results/final_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
              f"confusion_matrix_{os.path.basename(MODEL_PATH)[:-4]}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix → {save_path}")
    plt.show()