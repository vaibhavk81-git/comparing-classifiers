
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

The bank's problem is straightforward: they conducted telephone-based marketing campaigns to promote term deposits, but only about 11% of the target audience actually subscribed. 
That's a lot of wasted time and money. My goal was to build models that could:
1. Better identify customers likely to subscribe
2. Help the bank avoid calling people unlikely to convert
3. Make their marketing campaigns more cost-effective


---

## Dataset Summary

* **Source:** [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
* **Records:** 41,188
* **Features:** 20 independent variables + 1 target (`y`)
* **Target Variable:** `y` (binary) – *yes* (subscribed) or *no* (not subscribed)
* **Class Imbalance:** 89% non-subscribers vs. 11% subscribers

### Key Features

| Feature                                     | Description             |
| ------------------------------------------- | ----------------------- |
| `age`, `job`, `marital`, `education`        | Demographic information |
| `default`, `housing`, `loan`                | Financial profile       |
| `contact`, `month`, `day_of_week`           | Campaign details        |
| `campaign`, `pdays`, `previous`, `poutcome` | Past contact results    |
| `emp.var.rate`, `euribor3m`, `nr.employed`  | Economic indicators     |

---

## Methodology

### Modeling Approach

The notebook follows the **CRISP-DM framework**:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment and Recommendations

### Algorithms Compared

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Support Vector Machine (SVM)

### Model Tuning

Each model was fine-tuned using **GridSearchCV** and **cross-validation**, with **F1-score** as the primary metric due to class imbalance.

---

## Results Summary

| Model                              | Test Accuracy | Recall (Subscribed) | F1 (Subscribed) | Key Takeaway                                           |
| ---------------------------------- | ------------- | ------------------- | --------------- | ------------------------------------------------------ |
| **Logistic Regression (Balanced)** | **83.2%**     | **0.64**            | **0.46**        | Best recall; identifies the most potential subscribers |
| **KNN (Tuned)**                    | 88.4%         | 0.29                | 0.36            | Strong accuracy; moderate overfitting                  |
| **Decision Tree (Tuned)**          | 89.7%         | 0.29                | 0.39            | Interpretable, solid accuracy                          |
| **SVM (Tuned)**                    | 89.1%         | 0.27                | 0.35            | Stable, precise, lower recall                          |


### Conclusion

While Decision Tree and SVM achieved slightly higher overall accuracy, **Logistic Regression with class balancing** was the most aligned with the business goal — identifying the maximum number of likely subscribers.
It achieved the **highest recall (0.64)** and a balanced F1-score, ensuring more potential customers are reached.


---

## Key Insights

Feature importance analysis shows that customer history and timing are critical:

* **Previous Campaign Outcome (`poutcome`)** – People who said “yes” before are far more likely to subscribe again.
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
