
# Practical Application III: Comparing Classifiers

**UC Berkeley AI/ML Professional Certificate – Module 17.1**
**Author:** Vaibhav Khare

---

## Project Overview

This notebook explores how different machine-learning models can help a Portuguese bank improve the results of its term-deposit marketing campaigns.
The goal is to predict whether a client will subscribe to a deposit after a phone call and to use those predictions to target future campaigns more effectively.

Four supervised classifiers are built and compared:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Support Vector Machine (SVM)

---

## Business Problem

The bank runs phone campaigns to promote long-term deposits, but only about 11 percent of customers say “yes.”
By identifying which customers are most likely to subscribe, the bank can:

1. Focus calls on high-probability leads.
2. Reduce wasted time and cost.
3. Schedule campaigns when conditions are most favorable.

---

## Dataset

* **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
* **Rows:** 41,188 customers
* **Target:** `y` – *yes* (subscribed) / *no* (not subscribed)
* **Imbalance:** ≈ 89 % no  vs  11 % yes
* **Key features:** age, job, marital status, education, previous campaign results, economic indicators, and timing (month and day of week).

---

## Methodology

1. **Understand the data** – checked types, missing values, and class imbalance.
2. **Prepare the data** – encoded categorical fields and scaled numeric ones.
3. **Split the data** – 70 % train / 30 % test with stratified sampling.
4. **Model building** – trained Logistic Regression, KNN, Decision Tree, and SVM models.
5. **Hyperparameter tuning** – used GridSearchCV with cross-validation (scoring = F1).
6. **Evaluate models** – looked at accuracy, precision, recall, F1, and ROC-AUC.

---

## Results

| Model                          | Test Accuracy | Recall (Subscribed) | F1 (Subscribed) | Comment                           |
| ------------------------------ | ------------- | ------------------- | --------------- | --------------------------------- |
| Logistic Regression (Balanced) | **83.2 %**    | **0.64**            | **0.46**        | Catches the most real subscribers |
| KNN (Tuned)                    | 88.4 %        | 0.29                | 0.36            | Good accuracy but lower recall    |
| Decision Tree (Tuned)          | 89.7 %        | 0.29                | 0.39            | Interpretable; mild overfitting   |
| SVM (Tuned)                    | 89.1 %        | 0.27                | 0.35            | Stable but less sensitive         |

**Takeaway:**
Accuracy alone isn’t enough for an imbalanced dataset.
The **balanced Logistic Regression** model gives the best recall, meaning it identifies far more of the actual subscribers—exactly what the marketing team needs.

---

## Key Insights

Feature importance analysis shows that customer history and timing are critical:

* **Previous Campaign Outcome (`poutcome`)** – People who said “yes” before are far likelier to subscribe again.
* **Number of Contacts (`campaign`)** – Too many calls hurt conversion; quality beats quantity.
* **Economic Indicators (`emp.var.rate`, `euribor3m`, `nr.employed`)** – Stable conditions boost performance.
* **Month of Contact (`month`)** – March, June, September, and December perform best.
* **Days Since Last Contact (`pdays`)** – Recent contacts respond more positively.

In short: **who you call, when you call, and how recently you called** all matter more than call volume.

---

## Recommendations and Next Steps

### Short Term

1. Use the tuned Logistic Regression model with class balancing to score customers for the next campaign.
2. Start with a small pilot, focusing on high-probability leads to confirm lift in conversions.
3. Track recall, F1, and AUC monthly along with key economic variables.

### Long Term

1. Apply **SMOTE** or similar oversampling methods to handle imbalance more robustly.
2. Explore **ensemble models** (Random Forest, XGBoost) for stronger predictive power.
3. Integrate predictions into the bank’s **CRM**, giving agents live “likelihood-to-subscribe” scores.
4. **Retrain quarterly** using new campaign data to keep the model fresh.

---

## Visualizations

The notebook includes:

* Class-distribution chart (imbalance check)
* Correlation heatmap of numeric features
* Confusion matrix for Logistic Regression
* Feature-importance / coefficient plot
* ROC-AUC curves with labels and AUC values
* Bar chart comparing model Recall and F1

---

## Tools and Libraries

| Category            | Tools / Packages                                                   |
| ------------------- | ------------------------------------------------------------------ |
| **Language**        | Python 3                                                           |
| **Environment**     | Jupyter Notebook                                                   |
| **Libraries**       | pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn |
| **Version Control** | Git / GitHub                                                       |

---

## Deliverables

* `prompt_III.ipynb` – main notebook with full analysis
* `README.md` – project summary (this file)

---

## Acknowledgments

Dataset – UCI Machine Learning Repository
Course – UC Berkeley AI & Machine Learning Certificate
Tools – scikit-learn, seaborn, matplotlib

---

## Contact

**Vaibhav Khare**