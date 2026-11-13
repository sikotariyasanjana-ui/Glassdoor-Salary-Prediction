
# ============================================================
# GLASSDOOR SALARY PREDICTION - STREAMLIT APPLICATION
# ============================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Glassdoor Salary Prediction",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Glassdoor Salary Prediction")
st.markdown(
    "### Machine Learning Based Salary Prediction System"
)

st.write(
    "This application analyzes Glassdoor job data, "
    "performs exploratory data analysis and predicts "
    "average salaries using Random Forest Regression."
)


# ============================================================
# 2. DATASET LOADING
# ============================================================

def find_dataset():

    possible_paths = [
        Path("data/glassdoor_jobs.csv"),
        Path("data/glassdoor_jobs_cleaned.csv"),
        Path("glassdoor_jobs.csv"),
        Path("glassdoor_jobs_cleaned.csv")
    ]

    for file_path in possible_paths:
        if file_path.exists():
            return file_path

    return None


st.sidebar.title("Project Navigation")

uploaded_file = st.sidebar.file_uploader(
    "Upload Glassdoor Dataset",
    type=["csv"]
)

try:

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name

    else:

        dataset_path = find_dataset()

        if dataset_path is None:

            st.warning(
                "Dataset not found. Upload glassdoor_jobs.csv "
                "using the sidebar or place it inside the data folder."
            )

            st.stop()

        df = pd.read_csv(dataset_path)
        dataset_name = str(dataset_path)

except Exception as error:

    st.error(f"Dataset loading error: {error}")
    st.stop()


df.columns = df.columns.astype(str).str.strip()

st.sidebar.success("Dataset loaded successfully")


# ============================================================
# 3. SALARY DATA PREPROCESSING
# ============================================================

def extract_salary(value):

    if pd.isna(value):
        return np.nan

    text = str(value).replace(",", "").strip()

    if text.lower() in ["nan", "unknown", "n/a", "-1", ""]:
        return np.nan

    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return np.nan

    values = [float(x) for x in numbers[:2]]

    return sum(values) / len(values)


if "avg_salary" in df.columns:

    df["avg_salary"] = pd.to_numeric(
        df["avg_salary"],
        errors="coerce"
    )

elif "Salary Estimate" in df.columns:

    df["avg_salary"] = df["Salary Estimate"].apply(
        extract_salary
    )

elif "salary" in df.columns:

    df["avg_salary"] = pd.to_numeric(
        df["salary"],
        errors="coerce"
    )

else:

    st.error(
        "Salary column not found. Expected avg_salary, "
        "Salary Estimate or salary."
    )

    st.write("Available columns:", df.columns.tolist())
    st.stop()


df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=["avg_salary"])

df = df[df["avg_salary"] > 0]

df = df.drop_duplicates().reset_index(drop=True)


if len(df) < 10:

    st.error(
        "At least 10 valid salary records are required "
        "to train the Machine Learning model."
    )

    st.stop()


# ============================================================
# 4. FEATURE SELECTION
# ============================================================

target = "avg_salary"

excluded_columns = [
    "avg_salary",
    "salary",
    "Salary Estimate",
    "min_salary",
    "max_salary",
    "Min Salary",
    "Max Salary",
    "salary_estimate"
]

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
    and not col.lower().startswith("unnamed:")
    and not df[col].isna().all()
]

if len(feature_columns) == 0:

    st.error("No usable input features found in the dataset.")
    st.stop()


X = df[feature_columns].copy()
y = df[target].astype(float)

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]


# ============================================================
# 5. MACHINE LEARNING MODEL
# ============================================================

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    transformers = []

    if numeric_features:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2
            )
        )
    ])

    with st.spinner("Training Random Forest model..."):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

    st.sidebar.success("Model trained successfully")

except Exception as error:

    st.error(f"Model training error: {error}")
    st.stop()


# ============================================================
# 6. MODEL EVALUATION METRICS
# ============================================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Select Project Section",

    [
        "Home & Dataset",
        "Exploratory Data Analysis",
        "Model Evaluation",
        "Salary Prediction",
        "Download Results"
    ]
)


# ============================================================
# 8. HOME AND DATASET
# ============================================================

if page == "Home & Dataset":

    st.header("🏠 Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Job Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Total Features",
        len(feature_columns)
    )

    col3.metric(
        "Average Salary",
        f"{y.mean():,.2f}"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        info,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# ============================================================
# 9. EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "Exploratory Data Analysis":

    st.header("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Salary Distribution",
        "Category Analysis",
        "Missing Values",
        "Correlation"
    ])

    # Salary Distribution

    with tab1:

        st.subheader("Average Salary Distribution")

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.hist(
            df["avg_salary"],
            bins=25,
            edgecolor="black"
        )

        ax.set_xlabel("Average Salary")
        ax.set_ylabel("Frequency")
        ax.set_title("Salary Distribution")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Salary Outlier Analysis")

        fig, ax = plt.subplots(figsize=(9, 3))

        ax.boxplot(df["avg_salary"].dropna())

        ax.set_ylabel("Average Salary")
        ax.set_title("Salary Boxplot")

        st.pyplot(fig)
        plt.close(fig)

    # Category Analysis

    with tab2:

        st.subheader("Salary by Job Category")

        available_categories = [
            col for col in categorical_features
            if df[col].nunique(dropna=True) > 1
        ]

        if available_categories:

            selected_column = st.selectbox(
                "Select Category",
                available_categories
            )

            summary = (
                df.groupby(selected_column, dropna=False)
                ["avg_salary"]
                .mean()
                .sort_values(ascending=False)
                .head(15)
            )

            st.dataframe(summary.to_frame("Average Salary"))

            fig, ax = plt.subplots(figsize=(9, 5))

            summary.sort_values().plot(
                kind="barh",
                ax=ax
            )

            ax.set_xlabel("Average Salary")
            ax.set_title(
                f"Average Salary by {selected_column}"
            )

            st.pyplot(fig)
            plt.close(fig)

        else:

            st.info("No suitable categorical columns found.")

    # Missing Values

    with tab3:

        st.subheader("Missing Value Analysis")

        missing = df.isnull().sum()

        missing = missing[missing > 0].sort_values(
            ascending=False
        )

        if not missing.empty:

            st.bar_chart(missing)

            st.dataframe(
                missing.to_frame("Missing Values")
            )

        else:

            st.success(
                "No missing values found in the prepared dataset."
            )

    # Correlation

    with tab4:

        st.subheader("Numerical Feature Correlation")

        numeric_df = df.select_dtypes(
            include=["number"]
        )

        if numeric_df.shape[1] >= 2:

            correlation = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(10, 6))

            image = ax.imshow(
                correlation,
                cmap="coolwarm",
                aspect="auto",
                vmin=-1,
                vmax=1
            )

            ax.set_xticks(range(len(correlation.columns)))
            ax.set_yticks(range(len(correlation.columns)))

            ax.set_xticklabels(
                correlation.columns,
                rotation=90
            )

            ax.set_yticklabels(correlation.columns)

            fig.colorbar(image, ax=ax)

            ax.set_title("Correlation Heatmap")

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        else:

            st.info("Not enough numerical columns for correlation.")


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header("🤖 Random Forest Model Evaluation")

    st.caption(
        "Evaluation is performed using the 20% test dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("R² Score", f"{r2:.3f}")

    col2.metric("MAE", f"{mae:.3f}")

    col3.metric("MSE", f"{mse:.3f}")

    col4.metric("RMSE", f"{rmse:.3f}")

    results = pd.DataFrame({
        "Actual Salary": y_test.to_numpy(),
        "Predicted Salary": y_pred,
        "Absolute Error": np.abs(
            y_test.to_numpy() - y_pred
        )
    })

    st.subheader("Actual vs Predicted Results")

    st.dataframe(
        results.head(30),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.65
    )

    low = min(
        float(y_test.min()),
        float(np.min(y_pred))
    )

    high = max(
        float(y_test.max()),
        float(np.max(y_pred))
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--"
    )

    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Actual vs Predicted Salary")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 11. SALARY PREDICTION
# ============================================================

elif page == "Salary Prediction":

    st.header("💰 Salary Prediction")

    st.write(
        "Enter job details to estimate the average salary."
    )

    user_input = {}

    with st.form("salary_prediction_form"):

        for column in feature_columns:

            if column in numeric_features:

                values = pd.to_numeric(
                    X[column],
                    errors="coerce"
                )

                median_value = values.median()

                if pd.isna(median_value):
                    median_value = 0.0

                user_input[column] = st.number_input(
                    column,
                    value=float(median_value)
                )

            else:

                choices = (
                    X[column]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(20)
                    .index.tolist()
                )

                if choices:

                    user_input[column] = st.selectbox(
                        column,
                        choices
                    )

                else:

                    user_input[column] = st.text_input(
                        column
                    )

        submitted = st.form_submit_button(
            "Predict Salary",
            type="primary"
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_input],
            columns=feature_columns
        )

        try:

            prediction = model.predict(input_df)[0]

            st.success(
                f"Estimated Average Salary: {prediction:,.2f}"
            )

            st.info(
                "This is an estimated prediction based on "
                "historical dataset information."
            )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ============================================================
# 12. DOWNLOAD RESULTS
# ============================================================

elif page == "Download Results":

    st.header("📥 Download Prediction Results")

    results = pd.DataFrame({
        "Actual Salary": y_test.to_numpy(),
        "Predicted Salary": y_pred,
        "Absolute Error": np.abs(
            y_test.to_numpy() - y_pred
        )
    })

    st.dataframe(
        results,
        use_container_width=True
    )

    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction Results CSV",
        data=csv_data,
        file_name="glassdoor_salary_predictions.csv",
        mime="text/csv"
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Glassdoor Salary Prediction"
)

st.sidebar.caption(
    "Machine Learning Project | Python & Streamlit"
)

st.sidebar.caption(
    "Developed by Sanjana"
)
