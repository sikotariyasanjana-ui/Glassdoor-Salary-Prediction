# 💼 Glassdoor Salary Prediction Using Machine Learning

## 📌 Project Overview

The Glassdoor Salary Prediction project is a Machine Learning application developed using Python and Streamlit.

The main objective of this project is to analyze job-related information and predict the average salary based on different job features.

This project uses the **Random Forest Regression** algorithm to predict salaries and provides an interactive web application for users.

## 🎯 Project Objectives

- Analyze Glassdoor job-related data.
- Perform Exploratory Data Analysis (EDA).
- Handle missing values and preprocess the dataset.
- Visualize salary distributions and job categories.
- Build a Random Forest Regression model.
- Evaluate model performance using different metrics.
- Predict average salaries using user-entered job details.
- Download prediction results in CSV format.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10 | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Web application |
| Jupyter Notebook | Model development |

## 🤖 Machine Learning Algorithm

**Random Forest Regression**

Random Forest Regression is an ensemble learning algorithm that combines multiple decision trees to predict continuous numerical values.

It is used in this project to estimate average job salaries.

## 📊 Model Evaluation

The model is evaluated using:

1. R² Score
2. Mean Absolute Error (MAE)
3. Mean Squared Error (MSE)
4. Root Mean Squared Error (RMSE)

## 📁 Project Structure

```text
GLASSDOOR/
│
├── data/
│   └── glassdoor_jobs.csv
│
├── .vscode/
│
├── app.py
├── glassdoor EDA.ipynb
├── glassdoor ML.ipynb
├── README.md
└── requirements.txt
```

## ⚙️ Installation and Setup

### Step 1: Install Python

Install Python 3.10 or above.

### Step 2: Open Project in VS Code

Open the GLASSDOOR project folder in Visual Studio Code.

### Step 3: Open Terminal

Select Terminal → New Terminal.

### Step 4: Activate Virtual Environment

If your virtual environment is already created:

```bash
.\glassdoor_env\Scripts\activate
```

### Step 5: Install Required Libraries

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install project requirements:

```bash
python -m pip install -r requirements.txt
```

Alternatively:

```bash
python -m pip install streamlit pandas numpy matplotlib scikit-learn
```

## 🚀 How to Run the Project

Run the following command in the VS Code terminal:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

Local URL:

```text
http://localhost:8501
```

## 📈 Application Features

### 1. Home & Dataset
- Dataset preview
- Total job records
- Number of features
- Average salary
- Statistical summary

### 2. Exploratory Data Analysis
- Salary distribution
- Salary boxplot
- Category analysis
- Missing value analysis
- Correlation heatmap

### 3. Model Evaluation
- Random Forest Regression
- R² Score
- MAE
- MSE
- RMSE
- Actual vs Predicted Salary visualization

### 4. Salary Prediction
- Enter job-related details.
- Predict average salary.
- Display estimated salary.

### 5. Download Results
- View actual salary.
- View predicted salary.
- Download prediction results in CSV format.

## 📂 Dataset Requirements

Place your Glassdoor CSV dataset inside the `data` folder.

Expected dataset path:

```text
data/glassdoor_jobs.csv
```

The application also accepts a CSV uploaded through its sidebar.

The dataset must contain one of these salary columns:

- `avg_salary`
- `Salary Estimate`
- `salary`

Other job-related columns are used as prediction features.

## ▶️ Running Jupyter Notebooks

Open the following notebooks in Jupyter Notebook or VS Code:

1. `glassdoor EDA.ipynb`
2. `glassdoor ML.ipynb`

Run the notebook cells from top to bottom.

## 🌐 Deployment

This project can be deployed using Streamlit Community Cloud.

1. Upload the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Deploy the application.

GitHub: https://github.com/

Streamlit: https://streamlit.io/cloud

## 🔮 Future Enhancements

- Improve model accuracy through hyperparameter tuning.
- Add more regression algorithms.
- Improve the user interface.
- Add interactive salary comparison charts.
- Support additional job datasets.

## 👩‍💻 Project Information

**Project Name:** Glassdoor Salary Prediction

**Domain:** Machine Learning

**Algorithm:** Random Forest Regression

**Frontend:** Streamlit

**Language:** Python

**Developed by:** Sanjana

---

## ⭐ Conclusion

The Glassdoor Salary Prediction project demonstrates how Machine Learning can be used to analyze job-related data and estimate salaries.

The Streamlit application provides an interactive interface for dataset exploration, model evaluation and salary prediction.

This project helps users understand salary trends and make data-driven career decisions.