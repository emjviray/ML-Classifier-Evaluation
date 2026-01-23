# Machine Learning Classifier Performance Across Different Data Types (COGS 118A)

**Authors:** Emmanuel Viray III (eviray@ucsd.edu)  
**Affiliation:** Department of Cognitive Science, University of California, San Diego  
**Course:** COGS 118A — Supervised Machine Learning Algorithms  
**Term:** Fall 2025  

---

## 📌 Project Summary
This project evaluates and compares **five common supervised machine learning classifiers** across **four datasets**, with the goal of understanding how model performance changes depending on:

1. **Classifier type (algorithm behavior)**
2. **Dataset/data type**
3. **Train/test split ratio**
4. **Hyperparameter tuning strategy**
5. **Randomness and reliability across multiple trials**

To make the evaluation fair and consistent, all datasets were converted into **binary classification problems**, then tested using **Accuracy** and **F1 score** as the primary evaluation metrics.

Additionally, each experiment was repeated across **three independent trials** (different random seeds) to avoid misleading results due to randomness or a lucky/unlucky split. Each model was tuned using **3-fold stratified cross-validation** via `RandomizedSearchCV`.

This repo represents a full end-to-end ML workflow:
**preprocessing → hyperparameter search → repeated evaluation → reporting results**.

---

## 🔍 Research Question
**How do different supervised machine learning classifiers compare in performance (Accuracy + F1) across multiple datasets, and how does adjusting the train/test ratio influence generalization performance after hyperparameter tuning?**

---

## 🎯 Objectives
By the end of the project, we aimed to answer questions like:

- Which classifier is most **reliable across different data types**?
- Which model benefits the most from **additional training data**?
- When do Accuracy and F1 **disagree**, and what does that say about the dataset?
- Do some models **overfit more easily** depending on train/test ratio?
- How stable are results across repeated trials?

---

## 🧠 Classifiers Evaluated
This project compares five standard supervised learning classifiers:

### 1) Logistic Regression (LR)
- Strong baseline linear model
- Performs well on linearly separable data
- Often highly interpretable (weights/coefficients)

### 2) Decision Tree (DT)
- Nonlinear model that learns simple “if-then” rules
- Easy to interpret
- High risk of overfitting if not tuned

### 3) Random Forest (RF)
- Ensemble of decision trees
- Typically improves generalization
- Helps reduce variance and overfitting compared to a single decision tree

### 4) K-Nearest Neighbors (KNN)
- Instance-based model (no explicit training phase)
- Sensitive to feature scaling
- Performance depends heavily on k and distance metric

### 5) Support Vector Machine (SVM)
- Strong classification model that maximizes margin between classes
- Can handle nonlinear separation using kernels
- Effective but can be expensive computationally

---

## 📂 Datasets Used
All datasets were sourced from the **UCI Machine Learning Repository**, then transformed into binary tasks for consistent evaluation:

- **Heart Disease Dataset**
- **Adult Income Dataset**
- **Bank Marketing Dataset**
- **Wine Quality Dataset**

### Why multiple datasets?
Different datasets reflect different types of problems:
- balanced vs imbalanced classes  
- structured medical data vs economic/survey data  
- numeric feature distributions vs categorical-heavy datasets  
- easier vs harder classification boundaries  

This variety makes it possible to test whether a model performs well *universally*, or only under specific conditions.

---

## ⚙️ Data Preprocessing
Since this project compares across datasets, preprocessing was designed to ensure fairness:

### 1) Binary Classification Conversion
Some datasets contained multiple labels (ex: wine quality ratings). In those cases, labels were mapped into **two classes** (example: low vs high quality), ensuring consistent evaluation across all datasets.

### 2) Feature Preparation
Typical preprocessing steps included:
- Handling missing values where necessary
- Converting categories into numeric representations (encoding)
- Splitting features and targets into X and y

### 3) Scaling (Important for KNN + SVM)
Distance- and margin-based models such as **KNN** and **SVM** depend heavily on scaling. Feature normalization/standardization is required to prevent high-magnitude features from dominating the decision boundaries.

---

## 🔁 Experimental Design

### ✅ Train/Test Split Ratios
Each dataset was evaluated under three split configurations:

- **20/80** (small training data, large test set)
- **50/50** (balanced)
- **80/20** (large training data, smaller test set)

This allows analysis of:
- how much each classifier benefits from additional training data
- whether some classifiers fail under low training sizes
- generalization stability

---

## 🔧 Hyperparameter Optimization
Each classifier was tuned using:

- **RandomizedSearchCV**
- **3-fold Stratified Cross Validation**
- Hyperparameters specific to each model type

### Why RandomizedSearchCV?
Randomized search is computationally efficient and performs well in practice because:
- it explores the hyperparameter space more broadly
- avoids overfitting to a small grid
- scales better than exhaustive grid search

Stratified CV ensures each fold preserves class balance, which is especially important for imbalanced datasets.

---

## 🧪 Repeated Trials for Reliability
To reduce randomness in ML performance results, we ran:

✅ **3 independent trials per experiment**

Each trial changes the random seed so the results don’t depend on:
- a lucky split
- random initialization
- one unusual fold in cross-validation

Final performance values were averaged across trials for stability.

---

## 📊 Evaluation Metrics

### Accuracy
Measures total correct predictions:

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

Best when classes are balanced.

---

### F1 Score
Balances Precision and Recall:

\[
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

F1 is critical when:
- classes are imbalanced
- false positives vs false negatives matter differently
- accuracy gives misleading “high” performance

---

## 📈 Results & Interpretation (What to Look For)
This repo is designed for analysis of classifier behavior such as:

### Model behavior across splits
- Models like RF tend to become more stable and accurate with more training data
- KNN tends to struggle when training data is low
- Logistic Regression may perform surprisingly well depending on dataset linear separability

### Accuracy vs F1
- Sometimes accuracy stays high while F1 drops — indicating class imbalance effects
- F1 highlights how well the classifier handles minority classes

### Generalization trends
- A major goal is identifying which models overfit
- Comparing 80/20 vs 20/80 helps detect generalization failure

---

## 📌 Technologies Used
**Programming Language:** Python  
**Environment:** Jupyter Notebook  

### Core Libraries
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

### Scikit-learn components used
- `train_test_split`
- `StratifiedKFold`
- `RandomizedSearchCV`
- `accuracy_score`, `f1_score`
- classifiers: LR / DT / RF / KNN / SVM

---

## 🏗️ Suggested Repository Structure
If you want your repo to look extra clean for recruiters, this is a great structure:

