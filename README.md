# 🧪 Drug-Induced Autoimmunity (DIA) Prediction

## Problem

Some drugs can trigger autoimmune responses in certain individuals, a rare but potentially life-threatening side effect. The goal of this project is to build a machine learning pipeline to predict whether a drug is likely to cause drug-induced autoimmunity (DIA), based on its molecular and chemical descriptors.

---

## Data

- **Dataset:** A scientific dataset containing 196 molecular features (descriptors) per drug.
- **Target:** Binary classification — `1` if the drug causes DIA, `0` otherwise.
- **Source:** Dataset used under academic license. Referenced from Huang et al., 2025.

---

## Solution Overview

- Two classifiers were explored: **K-Nearest Neighbors (KNN)** and **Decision Trees**.
- Custom feature selection was done using **hill climbing**, optimizing for accuracy.
- **Recall** was prioritized due to the healthcare implications of false negatives.
- Hyperparameter tuning was performed for `K`, distance metrics, and tree depth.

---

## Modeling & Feature Selection

### K-Nearest Neighbors (KNN)

- **Scaling:** Data was standardized using Z-score normalization to handle feature magnitude differences (as KNN is distance-based).
- **Feature Selection:** A greedy hill climbing algorithm was implemented:
  - Start with a single random feature
  - Iteratively add features that improve **accuracy**
  - Selected set contained 62 features optimized for `K=3`
- **Bias Mitigation:** To avoid selection bias toward `K=3`, a second approach tested different feature sets per K-value (1–20), selecting the best-performing combination overall.
- **Final Model:** `K=5`, `weights='distance'`, `p=2` (Euclidean)

**Results:**
- **Accuracy:** 93%
- **Recall (Class 1 - DIA positive):** 77%
- **Precision (Class 1):** 100%
- **Interpretation:**  
  High precision indicates all predicted DIA-positive drugs are correct. However, recall indicates that 23% of actual DIA-positive drugs are missed — a concerning metric in healthcare.  
  After tuning weights to `'distance'`, recall improved by 4%, and we accepted a small trade-off in precision for better safety.

---

### Decision Trees

- No scaling required, as the model is not distance-based.
- The same hill climbing feature selection was applied.
- **Hyperparameters tuned:** primarily `max_depth`, using a grid search via `for` loops.
- **Best configuration:** `max_depth=7`

**Results:**
- **Accuracy:** 88%
- **Recall (Class 1 - DIA positive):** 57%
- **Precision (Class 1):** 94%
- **Interpretation:**  
  While accuracy is reasonable, the **recall is only slightly better than a coin flip** (57%). This means nearly half of the harmful drugs go undetected — an unacceptable risk in a medical application. Despite parameter tuning, Decision Trees underperformed compared to KNN in this task.

---

## Conclusion

- **Best Model:** KNN with `K=5` and `distance` weighting.
- **Key Takeaway:** In healthcare contexts, **recall must be prioritized** over raw accuracy or precision.
- **Final Decision:** Trade precision for improved recall to reduce the risk of releasing harmful drugs.

---

## Learnings

- Feature selection methods (hill climbing) need to align with the metric being optimized.
- Evaluation metrics should reflect **domain risk**, not just technical performance.
- Context-aware modeling decisions matter — especially in healthcare.

---

## 🧪 Limitations & Future Work

- 🔄 **Multicollinearity not addressed**  
  The molecular descriptors are mathematical equations and ranged features, many of which are highly correlated. Multicollinearity can distort distance calculations in KNN. While Decision Trees are more robust to this, removing correlated features or applying dimensionality reduction (e.g., PCA, VIF) could have improved performance for KNN.

- **Simplistic feature selection approach**  
  This project used a greedy hill climbing method focused on maximizing accuracy. In contrast, Huang et al. (2025) explored more sophisticated feature selection strategies that considered multiple evaluation metrics and domain relevance. Future work could explore:
  - Recursive Feature Elimination (RFE)
  - Mutual Information
  - Regularization-based methods (e.g., Lasso)
  - Model-based selection (e.g., XGBoost importances)

- **Class imbalance not handled**  
  The dataset was naturally imbalanced, with far fewer DIA-positive drugs. This can lead to biased models that perform poorly on the minority class. Techniques like the following could improve generalizability:
  - SMOTE (Synthetic Minority Oversampling)
  - Class weighting in model training
  - Undersampling majority class or ensemble rebalancing

- **Model generalizability not tested**  
  The model’s performance was only validated on a held-out test set. Cross-validation or external validation with a separate dataset would be necessary for production-level confidence.

