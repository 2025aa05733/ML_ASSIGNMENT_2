# ML Assignment 2 - Student Performance Prediction

## Links
- **GitHub**: (https://github.com/2025aa05733/ML_ASSIGNMENT_2)
- **Streamlit App**: https://2025aa05733-new.streamlit.app

---

## a. Problem Statement

The goal is to predict student final grades (G3) using machine learning. The grades range from 0 to 20, making this a multi-class classification problem with 17 classes.

I implemented 6 different ML models and compared their performance to find the best one.

---

## b. Dataset Description

| Info | Value |
|------|-------|
| Source | UCI ML Repository - Student Performance |
| Total Samples | 649 |
| Features | 32 |
| Target | G3 (final grade) |
| Classes | 17 unique grades |

**Features include:**
- Demographics: school, sex, age, address, family size
- Family: parent education (Medu, Fedu), parent jobs (Mjob, Fjob)
- Academic: study time, failures, school support, paid classes
- Lifestyle: activities, internet, romantic relationship, health
- Previous grades: G1, G2

---

## c. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.2923 | 0.8426 | 0.2910 | 0.2609 | 0.2483 | 0.2142 |
| Decision Tree | 0.4077 | 0.6529 | 0.3751 | 0.3553 | 0.3507 | 0.3420 |
| KNN | 0.2154 | 0.6485 | 0.2418 | 0.1830 | 0.1853 | 0.1168 |
| Naive Bayes | 0.1077 | 0.7577 | 0.2394 | 0.1868 | 0.0979 | 0.0740 |
| Random Forest | 0.4154 | 0.8971 | 0.3940 | 0.3412 | 0.3372 | 0.3456 |
| XGBoost | 0.4462 | 0.8683 | 0.3947 | 0.3876 | 0.3785 | 0.3833 |

**Best Model: XGBoost** with highest accuracy (0.4462) and MCC (0.3833)

---

## Model Observations

| ML Model Name | Observation |
|---------------|-------------|
| Logistic Regression | Linear model so it has trouble with complex patterns. Good AUC (0.84) means it ranks predictions well, but accuracy is low because class boundaries aren't linear. |
| Decision Tree | Better accuracy than logistic regression. Captures non-linear relationships but can overfit the training data. |
| KNN | Lowest performing model. The high dimensionality (32 features) and class imbalance makes it hard for KNN to find good neighbors. |
| Naive Bayes | Assumes features are independent which isn't true here. Has decent AUC but worst accuracy. Too simplistic for this dataset. |
| Random Forest | Best AUC score (0.897). Combines many trees so it's more robust than single decision tree. Good at handling mixed feature types. |
| XGBoost | Best overall performance. Gradient boosting builds trees sequentially to fix errors. Handles class imbalance well which is important since some grades have very few samples. |

---

## Project Structure

```
project/
├── app.py                 # streamlit web app
├── train_models.py        # training script
├── requirements.txt       # dependencies
├── README.md
├── data/
│   └── student_performance.csv
└── models/
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    ├── random_forest.py
    ├── xgboost_model.py
    └── model_comparison.csv   

```

---

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# train models
python train_models.py

# run streamlit app
streamlit run app.py
```

---

## Streamlit App Features

1. ✅ CSV file upload for test data
2. ✅ Model selection dropdown (6 models)
3. ✅ Evaluation metrics display
4. ✅ Confusion matrix visualization
5. ✅ Classification report

---

## Technologies

- Python 3.12
- scikit-learn
- XGBoost
- pandas, numpy
- Streamlit
- matplotlib, seaborn

---

BITS M.Tech AIML | ML Assignment 2 | Feb 2026
