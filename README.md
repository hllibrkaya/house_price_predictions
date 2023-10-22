# Melbourne Housing Price Prediction

This GitHub repository contains a machine learning project that focuses on predicting housing prices in Melbourne, Australia. The project involves data preprocessing, exploratory data analysis, feature engineering, and the development of a predictive model using various machine learning algorithms.

## Project Overview

The primary goal of this project is to build a predictive model for housing prices in Melbourne, based on a dataset from the file "Melbourne_housing_FULL.csv". The project includes the following key steps:

### Data Loading and Exploration

The project starts with loading the dataset and exploring its basic characteristics. This includes checking the shape, size, dimension, and obtaining summary statistics of the data.

### Data Preprocessing

- **Date Conversion:** The "Date" column is converted to a datetime format.
- **Missing Value Handling:** Rows with too many missing values are dropped, and missing values in numeric columns are filled with the mean, while missing values in categorical columns are filled with the mode.
- **Outlier Detection:** Outliers in numeric columns are detected using the Z-score method and replaced with the column's mean.

### Data Visualization

Visualizations are created to better understand the data. Box plots and histograms are used to visualize numeric features, while pair plots and a correlation heatmap offer insights into the relationships between variables.

### Feature Engineering

- **Categorical Encoding:** Categorical variables are one-hot encoded to prepare them for machine learning models.
- **Data Transformation:** The "Price" column is transformed using a logarithmic function.
- **Feature Scaling:** Features are standardized using the StandardScaler.

### Model Selection and Evaluation

Various machine learning models are evaluated using cross-validation to determine their performance in predicting housing prices. The following models are considered:

- Lasso
- Linear Regression
- Ridge
- Elastic Net
- K-Nearest Neighbors Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor

### Hyperparameter Tuning

Hyperparameters of the Random Forest Regressor are tuned using RandomizedSearchCV to find the best combination of hyperparameters for improved model performance.


## Model Results

After thorough analysis and evaluation, the best-performing model (Random Forest Regressor) has been selected. Here are the performance metrics for this model:

- **Mean Absolute Error (MAE):** 0.1455
- **Mean Squared Error (MSE):** 0.0401
- **Root Mean Squared Error (RMSE):** 0.2002
- **R-squared (R2) Score:** 0.8526

These metrics demonstrate the accuracy and reliability of the model in predicting housing prices in Melbourne. The R-squared score of 0.8526 indicates that the model explains a significant portion of the variance in the target variable.



## Repository Contents

The repository includes the following files:

- `Melbourne_housing_FULL.csv`: The dataset used in this project.
- `Melbourne_Housing_Price_Prediction.ipynb`: The Jupyter Notebook containing the Python code for this project.
- `README.md`: This document providing an overview of the project.

## Usage

To replicate or further develop this project, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the required libraries and dependencies installed, such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn.
3. Open the Jupyter Notebook, "Melbourne_Housing_Price_Prediction.ipynb," and run the cells to execute the code.


