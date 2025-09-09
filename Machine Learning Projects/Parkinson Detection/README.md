# Parkinson’s Disease Detection Using Vocal Biomarkers

## 📖 Introduction
This project presents the results of the analysis performed for detection of Parkinson’s Disease (PD) using vocal biomarkers.  
The workflow includes:
- **Exploratory Data Analysis (EDA)** to summarize statistical insights and check for missing data or anomalies
- **Data preprocessing** (outlier detection, feature selection, class balancing)
- **Model training** with three classifiers: Random Forest, Support Vector Machine (SVM), and XGBoost
- **Performance evaluation** using accuracy, AUC, and statistical tests

These results provide insights for early detection and potential control of Parkinson’s Disease.

---

## 🔍 Data Preprocessing & Transformation
- **Missing Values:** No missing values detected.
- **Outlier Detection:** Used Interquartile Range (IQR) method to cap extreme values.
- **Feature Selection:** Performed correlation analysis to remove redundant features and Recursive Feature Elimination (RFE) to select top predictors.
- **Class Balancing:** Applied SMOTE (Synthetic Minority Oversampling Technique) to handle imbalance.

---

## 📊 Handling Missing Values
- Checked across all 22 features — no missing values.
- Original data retained without imputation.

---

## 📈 Detecting and Handling Outliers (IQR Method)
- Boxplots used to detect outliers.
- Values outside `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR` replaced with bounds.
- Result: All features free from extreme outliers.


---

## 📉 Data Visualization
- Histograms: Distribution of numerical features.
- Bar Chart: Target variable distribution showing **48 healthy (0)** vs **147 Parkinson’s (1)** — significant imbalance.

**Figure 4:** Feature Distribution 
<img width="4500" height="3000" alt="Histogram" src="https://github.com/user-attachments/assets/62c04be1-1d0e-4ade-ab47-e7054ec5cf6d" />


**Figure 5:** Target Variable Distribution  
<img width="1800" height="1500" alt="barplots of target variable" src="https://github.com/user-attachments/assets/58a1fc76-6528-44ed-8872-489c7fa3baa3" />

---

## ⚖️ Handling Class Imbalance (SMOTE)
- Applied SMOTE to balance dataset.
- Prevents bias toward majority class and improves recall for minority class.

**Figure 6:** Balanced Class Distribution  
<img width="3600" height="1500" alt="after smote" src="https://github.com/user-attachments/assets/8f214cb4-b3d8-419d-a197-f78f5bf5e569" />

---

## 🔗 Correlation Analysis
- Generated correlation heatmap.
- Found strong correlations (> 0.8) among several features — risk of multicollinearity.
- Example: `MDVP: Shimmer` correlated 0.99 with `MDVP: Shimmer (dB)`.

**Figure 7:** Correlation Heatmap  
<img width="3600" height="3000" alt="heatmap" src="https://github.com/user-attachments/assets/2f1f7c30-360b-44de-be07-ceb8f6c917c2" />

---

## 🏆 Feature Selection
- Dropped one feature from each high-correlation pair.
- RFE with Random Forest selected **9 most important features**:
  - `MDVP_Fo_Hz_`
  - `MDVP_Fhi_Hz_`
  - `MDVP_Flo_Hz_`
  - `MDVP_Jitter_`
  - `MDVP_Shimmer`
  - `RPDE`
  - `DFA`
  - `spread2`
  - `D2`


---

## 🤖 Model Selection & Optimization

### Model Training
- Models: **SVM**, **Random Forest**, **XGBoost**
- Evaluated using **5-fold cross-validation**.
- Hyperparameter tuning with **GridSearchCV**.

### Performance Comparison

| Model        | Best Parameters | Best Score (CV) | Test Accuracy | Test AUC |
|--------------|----------------|-----------------|---------------|----------|
| SVM          | `{'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}` | 0.9277 | 1.0 | 1.0 |
| Random Forest| `{'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}` | 0.9574 | 1.0 | 1.0 |
| XGBoost      | `{'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1}` | 0.9617 | 0.9661 | 0.9977 |

---

## 📊 Model Evaluation

| Model         | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | Accuracy |
|---------------|--------------|--------------|------------|------------|--------------|--------------|----------|
| Random Forest | 1.0          | 1.0          | 1.0        | 1.0        | 1.0          | 1.0          | 1.0      |
| SVM           | 1.0          | 1.0          | 1.0        | 1.0        | 1.0          | 1.0          | 1.0      |
| XGBoost       | 1.0          | 0.9355       | 0.9333     | 1.0        | 0.9655       | 0.9667       | 0.9661   |

**Confusion Matrices:** Random Forest and SVM had perfect classification. 
**ROC Curves:** All three models showed near-perfect AUC.

---

## 📌 Feature Importance
- Random Forest feature importance highlighted the most predictive vocal biomarkers.



---

## 📊 Statistical Comparison
- Paired t-tests on F1-scores (5-fold CV) showed **no significant difference** between models (p > 0.05).

| Comparison               | t-statistic | p-value |
|--------------------------|-------------|---------|
| SVM vs Random Forest     | 0.277       | 0.795   |
| SVM vs XGBoost           | 0.474       | 0.66    |
| Random Forest vs XGBoost | 0.249       | 0.816   |

---

## 📂 Repository Structure

```
├── notebooks/          # Jupyter notebooks for analysis
├── src/                # Python scripts
├── figures/            # Saved plots & figures
├── README.md           # Project documentation
└── requirements.txt    # Dependencies
```


---

## 🛠️ Technologies Used
- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

---

## 📜 License
This project is licensed under the MIT License.

---

## 📬 Contact
**Your Name**  
GitHub: [@adroq](https://github.com/adroq-ai)  
Email: shehuroqeeb@gmail.com


