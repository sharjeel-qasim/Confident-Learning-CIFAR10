import numpy as np
import pandas as pd
import os
import pickle
from cleanlab.filter import find_label_issues
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy 

# ============================= CONFIG =============================
BASE_DIR = 'cl_results'
INPUT_DATA_DIR = BASE_DIR
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, 'prepared_data')
LOG_DIR = os.path.join(BASE_DIR, 'tracking_logs')

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================= PLOT FUNCTION =============================
# (Function definitions are kept standard)
def plot_issue_scores(pred_probs, is_issue_mask, noise_pct):
 scores = 1 - pred_probs.max(axis=1) # lower = more likely wrong
 plt.figure(figsize=(10, 6))
 sns.histplot(scores, bins=60, kde=True, color='lightblue', alpha=0.7, label='All samples')
 if is_issue_mask.any():
  thresh = scores[is_issue_mask].max()
  plt.axvline(thresh, color='red', ls='--', lw=2, label=f'Threshold = {thresh:.4f}')
 plt.title(f'Label Issue Score Distribution ({noise_pct}% Noise)')
 plt.xlabel('Issue Score (lower = more suspicious)')
 plt.ylabel('Count')
 plt.legend()
 plt.grid(True, alpha=0.3)
 plt.savefig(os.path.join(LOG_DIR, f'issue_scores_{noise_pct}p.png'), dpi=200, bbox_inches='tight')
 plt.close()

# --- Utility function to process and save a single noise level log ---
def process_and_save_log(issue_indices, noisy_labels, pred_probs, N, noise_pct):
    if len(issue_indices) > 0:
        suggested_labels = pred_probs.argmax(axis=1)
        
        mislabeled_df = pd.DataFrame({
            'Original_Index': issue_indices,
            'Noisy_Label': noisy_labels[issue_indices],
            'Noisy_Class': [CLASS_NAMES[noisy_labels[i]] for i in issue_indices],
            'Suggested_Label': suggested_labels[issue_indices],
            'Suggested_Class': [CLASS_NAMES[suggested_labels[i]] for i in issue_indices],
            'Confidence_Score': pred_probs[issue_indices].max(axis=1),
            'Action_Pruned_Models': 'REMOVED',
            'Action_Corrected_Models': 'FIXED → Suggested_Class'
        }).sort_values(by='Confidence_Score')

        log_path = os.path.join(LOG_DIR, f'label_issues_detailed_{noise_pct}p.csv')
        mislabeled_df.to_csv(log_path, index=False)
        print(f"   Detailed log saved → {log_path}")
        return mislabeled_df
    return None

# ============================= MAIN =============================
if __name__ == '__main__':
    cv_file = os.path.join(INPUT_DATA_DIR, 'cv_inputs.pkl')
    if not os.path.exists(cv_file):
        raise FileNotFoundError("Run step 1 first!")

    with open(cv_file, 'rb') as f:
        cv_results = pickle.load(f)

    print("\n" + "="*70)
    print("STEP 2: CONFIDENT LEARNING - FINAL BULLETPROOF VERSION")
    print("="*70)

    # --- PROCESS ALL THREE NOISE LEVELS (0%, 20%, 40%) ---
    noise_levels = [0, 20, 40]
    
    for noise_pct in noise_levels:
        noise_key = f"{noise_pct}_percent_noise"
        print(f"\n→ Processing {noise_key.replace('_', ' ').title()}")

        data = cv_results[noise_key]
        pred_probs = data['pred_probs']
        noisy_labels = data['noisy_labels']
        N = len(noisy_labels)

        # === 1. Get ranked list of suspected label issues ===
        issue_indices = find_label_issues(
            labels=noisy_labels,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence'
        )

        num_issues = len(issue_indices)
        is_issue = np.zeros(N, dtype=bool)
        is_issue[issue_indices] = True
        
        print(f" Confident Learning flagged {num_issues:,} issues ({num_issues/N*100:.2f}%)")

        # === 2. Plotting (Now includes the 0% case) ===
        plot_issue_scores(pred_probs, is_issue, noise_pct)

        # === 3. Create and save detailed log (Only if issues were found) ===
        mislabeled_df = process_and_save_log(issue_indices, noisy_labels, pred_probs, N, noise_pct)

        # === 4. Prepare corrected and pruned files ===
        if num_issues > 0:
            # Corrected labels (only for 20% and 40%, but this generalizes to 0% if mislabeled_df exists)
            corrected_labels = noisy_labels.copy()
            suggested_labels = mislabeled_df['Suggested_Label'].values # Use labels from the saved log
            corrected_labels[issue_indices] = suggested_labels 
            
            # Pruned indices
            clean_indices = np.arange(N)[~is_issue]
        else:
            # If 0 issues found (or for a perfectly clean 0% case)
            corrected_labels = noisy_labels.copy()
            clean_indices = np.arange(N)

        # === 5. Save all final files ===
        np.save(os.path.join(OUTPUT_DATA_DIR, f'pruned_indices_{noise_pct}p.npy'), clean_indices)
        np.save(os.path.join(OUTPUT_DATA_DIR, f'corrected_labels_{noise_pct}p.npy'), corrected_labels)
        np.save(os.path.join(OUTPUT_DATA_DIR, f'noisy_labels_{noise_pct}p.npy'), noisy_labels)

        print(f" All data files saved for {noise_pct}% noise\n")

    print("\nSUCCESS! All data prepared.")
    print(f" Data → {OUTPUT_DATA_DIR}")
    print(f" Logs → {LOG_DIR}")
    print("You can now run Step 3 and train your 9 models!")