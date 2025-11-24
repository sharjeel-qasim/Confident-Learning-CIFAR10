import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
import numpy as np
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy

NUM_CLASSES = 10
NUM_EPOCHS = 25
K_FOLDS = 4
BASE_DIR = 'cl_results'
os.makedirs(BASE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def create_model():
  """Initializes a ResNet-18 model configured for CIFAR-10."""
  net = timm.create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)
  net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  net.maxpool = nn.Identity()
  return net.to(DEVICE)


def simulate_noise(true_labels, noise_level):
  """Simulates asymmetric label noise (simplified for demonstration)."""
  noisy_labels = deepcopy(true_labels)
  noise_matrix = np.eye(NUM_CLASSES)
  
  # Asymmetric noise
  if noise_level == 20:
    # 20% of truck labels to be flipped to car and 20% of bird to airplane
    noise_matrix[9, 1] = 0.2; noise_matrix[9, 9] = 0.8
    noise_matrix[2, 0] = 0.2; noise_matrix[2, 2] = 0.8
  elif noise_level == 40:
    # 40% of cat to dog 40% of dog to frog, and 40% of bird to airplane
    noise_matrix[3, 5] = 0.4; noise_matrix[3, 3] = 0.6
    noise_matrix[5, 6] = 0.4; noise_matrix[5, 5] = 0.6
    noise_matrix[2, 0] = 0.4; noise_matrix[2, 2] = 0.6

  if noise_level > 0:
    for i in range(len(true_labels)):
      true_class = true_labels[i]
      noisy_labels[i] = np.random.choice(NUM_CLASSES, p=noise_matrix[true_class, :])
 
  return noisy_labels.tolist()


def train_and_predict_cv(trainset, noise_level):
  """Performs K-Fold CV, trains the model, and collects out-of-sample probabilities."""
  print(f"\n--- Starting {K_FOLDS}-Fold CV for {noise_level}% Noise ---")
  
  true_labels = np.array(trainset.targets) 
  
  trainset.targets = simulate_noise(true_labels, noise_level)
  all_noisy_labels = np.array(trainset.targets)
  all_pred_probs = np.zeros((len(trainset), NUM_CLASSES)) 
  kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
  
  for fold, (train_idx, val_idx) in enumerate(kf.split(trainset)):
     print(f"Fold {fold+1}/{K_FOLDS}...")
    
     train_subset = torch.utils.data.Subset(trainset, train_idx)
     val_subset = torch.utils.data.Subset(trainset, val_idx)
    
     trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
     valloader = torch.utils.data.DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
     net = create_model()
     criterion = nn.CrossEntropyLoss()
     optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) 
     # (K-1) Folds for 25 epochs
     for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1} Training"):
       net.train() 
       for inputs, targets in trainloader: 
         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
         optimizer.zero_grad()
         outputs = net(inputs) 
         loss = criterion(outputs, targets) 
         loss.backward() 
         optimizer.step() 
       scheduler.step()
     
     net.eval() 
     fold_probs = [] 
     with torch.no_grad(): 
       for inputs, _ in valloader: 
         inputs = inputs.to(DEVICE)
         outputs = net(inputs)
         probs = torch.softmax(outputs, dim=1) 
         fold_probs.append(probs.cpu().numpy()) 
     
     all_pred_probs[val_idx] = np.concatenate(fold_probs, axis=0) 
  return all_pred_probs, all_noisy_labels 

def plot_and_save_matrix(pred_probs, noisy_labels, noise_level):
    """
    Generates and saves a Confusion Matrix comparing Noisy Labels (True Y) 
    against the Model's Predictions (Predicted Y). This visualizes the input data quality.
    """
    predicted_labels = np.argmax(pred_probs, axis=1) 
    
    cm = confusion_matrix(noisy_labels, predicted_labels) 
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    plt.xlabel('Predicted Label (Model\'s Unbiased Guess)') 
    plt.ylabel('Noisy Given Label')
    plt.title(f"CV Input Matrix: {noise_level}% Noise (Noisy Label vs. Model Guess)")
    plt.tight_layout()
    
    filename = os.path.join(BASE_DIR, f'cv_input_matrix_{noise_level}p.png')
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {filename}")

if __name__ == '__main__':
  transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  
  cifar10_train_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
  results = {}
  
  # 0% Noise
  pp_0, y_0 = train_and_predict_cv(deepcopy(cifar10_train_clean), 0)
  results['0_percent_noise'] = {'pred_probs': pp_0, 'noisy_labels': y_0}
  plot_and_save_matrix(pp_0, y_0, 0)
  
  # 20% Noise
  pp_20, y_20 = train_and_predict_cv(deepcopy(cifar10_train_clean), 20)
  results['20_percent_noise'] = {'pred_probs': pp_20, 'noisy_labels': y_20}
  plot_and_save_matrix(pp_20, y_20, 20) 
  
  # 40% Noise
  pp_40, y_40 = train_and_predict_cv(deepcopy(cifar10_train_clean), 40)
  results['40_percent_noise'] = {'pred_probs': pp_40, 'noisy_labels': y_40}
  plot_and_save_matrix(pp_40, y_40, 40)
  
  filename = os.path.join(BASE_DIR, 'cv_inputs.pkl')
  
  with open(filename, 'wb') as f: 
    pickle.dump(results, f)
  
  print("\n" + "="*60)
  print("STEP 1: INPUT GENERATION COMPLETE.")
  print(f"CV predicted probabilities saved to: {filename}")
  print("="*60)