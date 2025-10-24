# Heart Disease Prediction Analysis

This repository contains a Jupyter Notebook that performs an analysis on the heart disease dataset to predict the presence of heart disease. The analysis includes data preprocessing, model training with and without Principal Component Analysis (PCA), and a comparison of model performance.

## Dataset

The dataset used in this analysis is the "Heart Failure Prediction" dataset, available on Kaggle (credits: https://www.kaggle.com/fedesoriano/heart-failure-prediction). It contains various health metrics and a target variable indicating the presence or absence of heart disease.

## Analysis Steps

The notebook follows these steps:

1.  **Load Data:** The `heart.csv` dataset is loaded into a pandas DataFrame.
2.  **Handle Outliers:** Outliers in the numerical columns are removed using the Z-score method (with a threshold of 3).
3.  **Encode Categorical Features:** Categorical features are converted into numerical representations using one-hot encoding.
4.  **Scale Data:** Numerical features are standardized using `StandardScaler`.
5.  **Model Training (without PCA):** Classification models (SVM, Logistic Regression, Random Forest) are trained and evaluated on the preprocessed data.
6.  **Apply PCA:** Principal Component Analysis (PCA) is applied to reduce the dimensionality of the data while retaining 95% of the variance.
7.  **Model Training (with PCA):** The same classification models are trained and evaluated on the PCA-transformed data.
8.  **Compare Results:** The performance of the models with and without PCA is compared based on accuracy.

## Findings

*   Outlier removal using the Z-score method resulted in the removal of 19 rows.
*   Categorical features were successfully encoded.
*   Numerical features were scaled.
*   Without PCA, SVM and Logistic Regression achieved an accuracy of approximately 88.89%, while Random Forest achieved approximately 88.33%.
*   Applying PCA (retaining 95% variance) reduced the dimensionality from 15 to 10 components.
*   With PCA, SVM accuracy remained at approximately 88.89%.
*   With PCA, Logistic Regression accuracy decreased to 85.00%.
*   With PCA, Random Forest accuracy decreased to 83.33%.
*   The SVM model achieved the highest accuracy in both scenarios.

## Conclusion

The analysis shows that for this dataset, applying PCA did not improve the accuracy of the tested models and even reduced the performance of Logistic Regression and Random Forest. The SVM model performed best, achieving an accuracy of approximately 88.89% both with and without PCA.

## How to Use

1.  Clone this repository.
2.  Ensure you have the necessary libraries installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).
3.  Download the `heart.csv` dataset and place it in the appropriate directory.
4.  Run the Jupyter Notebook to reproduce the analysis.
