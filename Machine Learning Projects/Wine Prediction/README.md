
# Wine Quality Prediction using Random Forest Classifier

Welcome to the **Wine Quality Prediction** project! This repository showcases how to build a machine learning model to predict the quality of wine using the **Random Forest Classifier**. The project includes exploratory data analysis, model training, evaluation, and deployment via a user-friendly Gradio interface.

![Wine Prediction Banner](https://github.com/user-attachments/assets/acc8bacb-c9ec-43e8-91a9-3f9376464a8b)
</br>


![Wine Prediction Banner](https://github.com/user-attachments/assets/e31f11fe-29d7-4848-92dc-994790cb11e8)
</br>

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Features](#features)
6. [Usage](#usage)
7. [Results](#results)
8. [Visualization](#visualization)
9. [License](#license)

---

## Overview

This project predicts the quality of red wine based on its physicochemical properties. The dataset consists of attributes such as acidity, sugar, pH levels, and alcohol concentration. Using **Random Forest Classifier**, we achieve high accuracy in predicting whether a wine is of good quality or not.

### Project Statement

![Project Statement](https://github.com/user-attachments/assets/e5dc1ebf-fcc9-44a4-8867-ec6abab0e9c0)
</br


### Work Flow

![Work Flow](https://github.com/user-attachments/assets/997ec794-d8b7-4ea0-8440-7594360aca77)
</br>



### Key Objectives:
- Perform **Exploratory Data Analysis (EDA)**.
- Build a **Random Forest Classifier** for prediction.
- Deploy a **Gradio Interface** for real-time predictions.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/AdMub/Data-Science-Project.git
   cd wine-quality-prediction
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook
   ```

---

## Project Structure

```
Wine Quality Prediction/
├── data/
│   └── winequality-red.csv  # Dataset
├── notebooks/
│   └── WineQualityPrediction.ipynb  # Jupyter Notebook
├── app.py  # Gradio Deployment Script
├── requirements.txt  # Required Libraries
└── README.md  # Project Documentation
```

---

## Dataset

The dataset used for this project is the [Red Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality), which contains:

- **1599 samples**
- **11 physicochemical features**
- **1 target feature** (`quality`)

### Features:
- `fixed acidity`, `volatile acidity`, `citric acid`
- `residual sugar`, `chlorides`, `free sulfur dioxide`
- `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`
- `quality` (target variable, binary classification: Good or Not Good)

---

## Features

1. **EDA**: Data visualization and correlation analysis.
2. **Model Building**: Random Forest Classifier.
3. **Evaluation**: Accuracy, confusion matrix, and classification report.
4. **Deployment**: Gradio app for user interaction.

---

## Usage

### 1. Run the Jupyter Notebook:
- Open `WineQualityPrediction.ipynb` to explore the data, train the model, and evaluate its performance.

### 2. Deploy Gradio Interface:
   ```bash
   python app.py
   ```
   ![Gradio Interface Example](https://github.com/user-attachments/assets/788fb7ae-ceb1-442d-a97d-68952e599617)


---

## Results

### Model Performance:
| Metric            | Training Data | Test Data |
|-------------------|---------------|-----------|
| **Accuracy**      | 100%          | 91%       |
| **Precision**     | 93%           | 64%       |
| **Recall**        | 97%           | 47%       |
| **F1-Score**      | 95%           | 54%       |

---

## Visualization

### Example Correlation Heatmap:
![Correlation Heatmap](https://github.com/user-attachments/assets/f1eaa9bc-6378-408e-86fc-016e3a21bf43)
</br>

### Example Bar Plot:

-- Volatile Acidity vs Quality

![Bar Plot](https://github.com/user-attachments/assets/663f86e7-98b3-43b6-844f-8d1b25a76eff)
</br>

-- Citric Acid vs Quality

![Bar Plot](https://github.com/user-attachments/assets/fe8a22ed-bc35-4dd0-8193-aac1c31f8cfc)
</br>
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Contributors

- [AdMub](https://github.com/AdMub)

Feel free to contribute or raise an issue if you encounter any problems!
