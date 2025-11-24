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

# --- Configuration ---
NUM_CLASSES = 10 # CIFAR-10 has 10 distinct classes.
NUM_EPOCHS = 25 # Training epochs per fold (Total 100 epochs of work per noise level).
K_FOLDS = 4 # Using 4-fold CV to ensure 100% of data is predicted out-of-sample.
BASE_DIR = 'cl_results' # Name of the main directory to save all outputs and logs.
os.makedirs(BASE_DIR, exist_ok=True) # Creates the results folder if it doesn't exist.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Automatically uses GPU if available.
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # Human-readable class names for plotting.


def create_model():
  """Initializes a ResNet-18 model configured for CIFAR-10."""
  # Creates the ResNet-18 architecture, ensuring it's not pre-trained on ImageNet.
  net = timm.create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)
  # Adjusts the first convolutional layer for the smaller 32x32 CIFAR images.
  net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  # Replaces the standard max-pooling layer with an identity function (often done for CIFAR).
  net.maxpool = nn.Identity()
  return net.to(DEVICE) # Sends the model weights to the GPU/CPU device.


def simulate_noise(true_labels, noise_level):
  """Simulates asymmetric label noise (simplified for demonstration)."""
  noisy_labels = deepcopy(true_labels) # Creates a mutable copy of the labels to modify.
  noise_matrix = np.eye(NUM_CLASSES) # Starts with an identity matrix (0% noise).
  
  # Asymmetric noise examples (following patterns often seen in CIFAR)
  if noise_level == 20:
    # Sets 20% of 'truck' labels to be flipped to 'car', and 20% of 'bird' to 'airplane'.
    noise_matrix[9, 1] = 0.2; noise_matrix[9, 9] = 0.8
    noise_matrix[2, 0] = 0.2; noise_matrix[2, 2] = 0.8
  elif noise_level == 40:
    # Sets 40% of 'cat' to 'dog', 40% of 'dog' to 'frog', and 40% of 'bird' to 'airplane'.
    noise_matrix[3, 5] = 0.4; noise_matrix[3, 3] = 0.6
    noise_matrix[5, 6] = 0.4; noise_matrix[5, 5] = 0.6
    noise_matrix[2, 0] = 0.4; noise_matrix[2, 2] = 0.6

  if noise_level > 0:
    for i in range(len(true_labels)): # Loops through every image in the training set.
      true_class = true_labels[i] # Gets the original (true) label index.
      # Probabilistically draws a new label based on the row in the noise matrix.
      noisy_labels[i] = np.random.choice(NUM_CLASSES, p=noise_matrix[true_class, :])
 
  return noisy_labels.tolist() # Returns the list of intentionally noisy labels.


def train_and_predict_cv(trainset, noise_level):
  """Performs K-Fold CV, trains the model, and collects out-of-sample probabilities."""
  print(f"\n--- Starting {K_FOLDS}-Fold CV for {noise_level}% Noise ---") # Tracks progress.
  
  true_labels = np.array(trainset.targets) # Stores the original clean labels temporarily.
  
  # 1. Apply simulated noise to the targets
  trainset.targets = simulate_noise(true_labels, noise_level) # Overwrites the dataset labels with noisy ones.
  all_noisy_labels = np.array(trainset.targets) # Stores the final noisy labels (Y-tilde). 
  # Storage array: 50,000 images x 10 classes
  all_pred_probs = np.zeros((len(trainset), NUM_CLASSES)) 
  # Sets up the KFold splitter for 4 non-overlapping groups.
  kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
  
  # --- CV Loop: Executes 4 times, once for each fold ---
  for fold, (train_idx, val_idx) in enumerate(kf.split(trainset)):
     print(f"Fold {fold+1}/{K_FOLDS}...") # Tracks which fold is running.
    
     # Creates subsets using indices: 75% for training, 25% for prediction.
     train_subset = torch.utils.data.Subset(trainset, train_idx)
     val_subset = torch.utils.data.Subset(trainset, val_idx)
    
     # Creates data loaders for efficient data processing.
     trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
     valloader = torch.utils.data.DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
     # Initializes a brand new ResNet model for this fold (no previous weights).
     net = create_model()
     criterion = nn.CrossEntropyLoss() # Standard loss function for multi-class classification.
     optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) # Optimizer settings.
     # Scheduler controls the learning rate decay over 25 epochs.
     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) 
     # Train on (K-1) Folds for 25 epochs
     for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1} Training"):
       net.train() # Sets model to training mode.
       for inputs, targets in trainloader: # Standard training loop iterates over batches.
         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
         optimizer.zero_grad() # Clears old gradients.
         outputs = net(inputs) # Forward pass (gets predictions).
         loss = criterion(outputs, targets) # Calculates the loss against noisy targets.
         loss.backward() # Backpropagation (calculates new gradients).
         optimizer.step() # Updates model weights.
       scheduler.step() # Adjusts the learning rate. 
     # Predict on the Held-Out Fold (Out-of-Sample Prediction)
     net.eval() # Sets model to evaluation mode (disables dropout, etc.).
     fold_probs = [] # List to collect predictions for this 25% fold.
     with torch.no_grad(): # Disable gradient calculations (saves time and memory).
       for inputs, _ in valloader: # Iterates over the held-out 25% data.
         inputs = inputs.to(DEVICE)
         outputs = net(inputs)
         # Convert raw outputs to predicted probabilities (softmax).
         probs = torch.softmax(outputs, dim=1) 
         fold_probs.append(probs.cpu().numpy()) # Stores the probabilities. 
     # Stores the predictions into the main array at the correct indices.
     all_pred_probs[val_idx] = np.concatenate(fold_probs, axis=0) 
  return all_pred_probs, all_noisy_labels # Returns the two key inputs for Cleanlab.

def plot_and_save_matrix(pred_probs, noisy_labels, noise_level):
    """
    Generates and saves a Confusion Matrix comparing Noisy Labels (True Y) 
    against the Model's Predictions (Predicted Y). This visualizes the input data quality.
    """
    # 1. Determine the Model's predictions (argmax of predicted probabilities)
    predicted_labels = np.argmax(pred_probs, axis=1) # Gets the index of the highest probability.
    
    # 2. Compute the Confusion Matrix
    cm = confusion_matrix(noisy_labels, predicted_labels) # Calculates the matrix: (Given Label vs. Model Guess).
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    # Uses seaborn for a professional, colorful heatmap visualization.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    # Labeling
    plt.xlabel('Predicted Label (Model\'s Unbiased Guess)') # The model's confident prediction (y* estimate).
    plt.ylabel('Noisy Given Label') # The label the model actually trained on (y_tilde).
    plt.title(f"CV Input Matrix: {noise_level}% Noise (Noisy Label vs. Model Guess)")
    plt.tight_layout()
    
    # Saving
    filename = os.path.join(BASE_DIR, f'cv_input_matrix_{noise_level}p.png')
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close() # Closes the plot to free up memory.
    print(f"Visualization saved to: {filename}")


# --- Main Execution ---
if __name__ == '__main__':
  # Standard image transformations for consistency.
  transform = transforms.Compose([
     transforms.ToTensor(),
     # Normalize the image pixel values with standard CIFAR-10 means and std devs.
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  
  # Load CIFAR-10 training set (clean labels)
  cifar10_train_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
  results = {}
  
  # --- 0% Noise
  pp_0, y_0 = train_and_predict_cv(deepcopy(cifar10_train_clean), 0)
  results['0_percent_noise'] = {'pred_probs': pp_0, 'noisy_labels': y_0}
  plot_and_save_matrix(pp_0, y_0, 0) # Saves the 0% confusion matrix image.
  
  # --- 20% Noise ---
  pp_20, y_20 = train_and_predict_cv(deepcopy(cifar10_train_clean), 20)
  results['20_percent_noise'] = {'pred_probs': pp_20, 'noisy_labels': y_20}
  plot_and_save_matrix(pp_20, y_20, 20) # Saves the 20% confusion matrix image. 
  
  # --- 40% Noise ---
  pp_40, y_40 = train_and_predict_cv(deepcopy(cifar10_train_clean), 40)
  results['40_percent_noise'] = {'pred_probs': pp_40, 'noisy_labels': y_40}
  plot_and_save_matrix(pp_40, y_40, 40) # Saves the 40% confusion matrix image. 
  
  # Save results
  filename = os.path.join(BASE_DIR, 'cv_inputs.pkl')
  
  # Uses pickle to serialize and save the large results dictionary efficiently.
  with open(filename, 'wb') as f: 
    pickle.dump(results, f)
  
  print("\n" + "="*60)
  print("STEP 1: INPUT GENERATION COMPLETE.")
  print(f"CV predicted probabilities saved to: {filename}")
  print("="*60)