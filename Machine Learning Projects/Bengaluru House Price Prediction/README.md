# 🧠 Bengaluru House Price Prediction

This project predicts house prices in **Bengaluru, India** using **machine learning**.  
It involves data cleaning, feature engineering, model training, and deployment with **Flask**.

---

## 🎯 Objective
To build a regression model that estimates house prices based on:
- 📍 Location  
- 📏 Total square feet  
- 🛁 Number of bathrooms  
- 🏘️ BHK (Bedrooms, Hall, Kitchen)

---

## 📊 Data Science Workflow

1. **Data Cleaning**
   - Removed missing values and duplicates  
   - Converted `total_sqft` to numeric values  
   - Handled outliers and unrealistic price ranges  

2. **Feature Engineering**
   - One-hot encoded categorical locations  
   - Created `price_per_sqft`  
   - Grouped rare locations  

3. **Modeling**
   - Algorithms tested: Linear Regression, Lasso, Decision Tree.  
   - Evaluated using **Score** 

4. **Model Deployment**
   - Final model saved as `home_prices_model.pickle`  
   - Flask app (`app.py`) serves predictions through a web interface  

---

## 🧠 Tech Stack

| Category | Tools/Libraries |
|-----------|----------------|
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deployment | Flask |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

- house_price_prediction.ipynb          # Data cleaning and model training
- home_prices_model.pickle              # Trained ML model
- columns.json                          # Feature information
- app.py                                # Flask web app
- index.html                            # Frontend page
- style.css                             # CSS styling
- script.js                              # JavaScript file

---

## ⚙️ How to Run

```bash
git clone https://github.com/your-github-username/Bengaluru-House-Price-Prediction.git
cd Bengaluru-House-Price-Prediction
pip install -r requirements.txt
python app.py
```


Open your browser and go to http://127.0.0.1:5000/
Enter property details and click Predict Price to get the result.


🔍 **Key Insights**
Location has the strongest effect on price.
BHK and area show non-linear relationships with price.
The model balances accuracy with interpretability.


👨‍💻 **Author**
Roqeeb Adisa
🎓 MSc. Statistics, University of Ibadan, Nigeria
Shehuroqeeb@gmail.com
