# 📦 Product Demand Prediction

A beginner-friendly data science project that predicts product demand using the Superstore Sales dataset.

---

## Project Goal

The goal of this project is to **predict product demand (Quantity)** based on sales data features like sales amount, discount, and profit.

> **What is "Demand"?** In this project, we use **Quantity** (number of units sold) as a measure of product demand.

---

## 📊 Dataset Used

**Sample - Superstore.csv**

This dataset contains sales data for a fictional retail store, including:
- Sales information
- Product quantities sold
- Discounts applied
- Profit margins
- Customer and order details

---

## 🚀 How to Run This Project

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Run the Main Script

This will train the model and save it:

```bash
python main.py
```

### Step 3: Launch the Prediction App

```bash
streamlit run streamlit_app.py
```

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `main.py` | Main script for data processing, model training, and evaluation |
| `streamlit_app.py` | Simple web app for making predictions |
| `requirements.txt` | List of Python libraries needed |
| `README.md` | This file - project documentation |

---

## 🔧 Steps Performed

### 1️⃣ Data Cleaning
- Checked for missing values
- Removed duplicate rows
- Ensured data quality

### 2️⃣ Exploratory Data Analysis (EDA)
Created visualizations to understand the data:
- **Distribution of Quantity** - Shows how demand is spread across products
- **Sales vs Quantity** - Relationship between sales amount and units sold
- **Discount vs Quantity** - How discounts affect demand
- **Correlation Heatmap** - Shows relationships between numeric features

All plots are saved in the `plots/` folder.

### 3️⃣ Feature Engineering
Created a new feature to help the model learn better:

```
price_per_unit = Sales / Quantity
```

This feature represents the average price per unit sold.

### 4️⃣ Model Training
Trained two simple regression models:
- **Linear Regression** - A simple, interpretable model
- **Random Forest Regressor** - An ensemble model for better accuracy

**Features used:**
- Sales
- Discount
- Profit
- price_per_unit

**Target variable:** Quantity (product demand)

### 5️⃣ Model Evaluation
Compared models using:
- **MAE (Mean Absolute Error)** - Average prediction error
- **R² Score** - How well the model explains the data (0 to 1, higher is better)

The best performing model is saved as `model.pkl`.

### 6️⃣ Prediction
Use the Streamlit app to enter:
- Sales amount
- Discount percentage
- Profit margin

And get a predicted demand (quantity)!

---

## 📈 Sample Output

When you run `main.py`, you'll see output like:

```
============================================================
PRODUCT DEMAND PREDICTION PROJECT
============================================================

1. LOADING DATASET...
----------------------------------------
Dataset loaded successfully!
Shape: 9994 rows, 21 columns

...

==================================================
Model                     MAE          R2 Score    
==================================================
Linear Regression         2.3456       0.1234      
Random Forest             1.8765       0.4567      
==================================================

Best Model: Random Forest (higher R2 score)
```

---

## 💡 Key Learnings

This project demonstrates:
- How to load and clean real-world sales data
- Basic exploratory data analysis with visualizations
- Simple feature engineering
- Training and comparing machine learning models
- Creating an interactive prediction interface

---

## 🛠️ Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning
- **Streamlit** - Web app framework
- **Joblib** - Model serialization

---

## 📌 Notes for Beginners

- **Quantity = Demand**: In this project, we use the number of units sold (Quantity) as our measure of product demand.
- **Simple Approach**: This project intentionally avoids complex ML pipelines to make it easy to understand.
- **Feature Engineering**: Creating `price_per_unit` helps the model understand the relationship between sales and quantity.

