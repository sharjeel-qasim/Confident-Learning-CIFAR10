# 🎓 Confident Learning (CL) on CIFAR-10

This repository implements and evaluates **Confident Learning (CL)** to detect and correct label errors in the CIFAR-10 dataset.  
The project follows a rigorous **data-centric pipeline**, demonstrating that cleaning training data yields significant, measurable improvements over standard training.

---

## 🔬 Methodology: Data-Centric Validation

The pipeline consists of **three sequential steps**:

1. **Input Generation (CV)**  
2. **Error Detection (CL)**  
3. **Final Evaluation (Sequential Training)**  

### Key Components

- **Prediction Engine:**  
  A customized **ResNet-18** model is used across all steps.

- **Noise Conditions:**  
  - **0% Intrinsic Noise**  
  - **20% Asymmetric Label Noise**  
  - **40% Asymmetric Label Noise**

- **Core Algorithm:**  
  The project uses the `cleanlab` library to compute the **Confident Joint**  
  \( C_{\tilde{y}, y^{*}} \),  
  which statistically quantifies label errors.

---

## 📁 Repository Structure

| File | Purpose | Notes |
|------|---------|-------|
| `1_generate_inputs_cv.py` | **Step 1: Input Generation** | Runs **4-Fold CV** to generate unbiased predicted probabilities \( \hat{P}_{k,i} \). |
| `2_detect_and_prepare_data.py` | **Step 2: Cleaning & Logging** | Performs CL, logs detected errors, and creates final datasets. |
| `3_sequential_training.py` | **Step 3: Final Evaluation** | Trains 9 models for **100 epochs** each with checkpointing. |
| `4_testing_confusion_matrics.py` | **Step 4: Confusion Matrix** | Produces confusion matrixes using testing data and saved models |
| `cl_results/` | Output Directory | Contains saved models, logs, metrics, and cleaned datasets. |

---

## 🚀 How to Run the Project

### Install Dependencies

```bash
pip install torch torchvision timm numpy pandas scikit-learn tqdm matplotlib seaborn
