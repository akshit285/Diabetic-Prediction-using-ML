# Diabetic-Prediction-using-ML
Comparative analysis of various Machine Learning Classification algorithms on the diabetic prediction dataset.

# Diabetic Prediction ML Project Documentation
### Introduction
This documentation provides an overview and step-by-step explanation of a Machine Learning project focused on predicting diabetes using various classification models. The project is based on a dataset obtained from Kaggle and was implemented using a Jupyter Notebook.

### Table of Contents
#### 1. Project Overview
- Objective
- Dataset Information
- Features and Target
- Libraries Used
#### 2. Data Preprocessing
- Loading the Dataset
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Data Visualization
#### 3. Data Splitting
- Train-Test Split
- Feature Scaling (Standardization)
#### 4. Model Implementation
- Random Forest Classifier
- Support Vector Machine (SVM)
  - Linear Kernel
  - RBF Kernel
  - Polynomial Kernel
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Naive Bayes
- XGBoost Classifier
#### 5. Model Evaluation
- Confusion Matrix
- Accuracy Score
- Precision, Recall, and F1 Score
- Heatmap Visualization
#### 6. Model Comparison and Selection
- Evaluation Metrics Comparison
- Selecting the Best Model
#### 7. Conclusion
- Summary of Results
- Future Enhancements

### 1. Project Overview
#### Objective
The main goal of this project is to develop accurate machine learning models that can predict whether a patient has diabetes or not based on various medical features. This will assist healthcare professionals in making early diagnoses and interventions.

#### Dataset Information
The dataset used in this project consists of medical information from patients, including features such as pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age. The target variable is 'Outcome', indicating whether a patient has diabetes (1) or not (0).

#### Libraries Used
The following Python libraries were utilized in this project:

- NumPy
- Matplotlib
- Pandas
- Seaborn
- Plotly Express
- Scikit-learn
- XGBoost

### 2. Data Preprocessing
In this section, the dataset is loaded, analyzed using exploratory data analysis (EDA), cleaned, and visualized to gain insights into the data distribution and relationships.

### 3. Data Splitting
The dataset is split into training and testing sets using the train_test_split function from scikit-learn. Additionally, feature scaling is applied to standardize the feature values.

### 4. Model Implementation
Multiple classification models are implemented and trained using the training data. The models include:

- Random Forest Classifier
- Support Vector Machine (SVM) with different kernels (linear, RBF, and polynomial)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Naive Bayes
- XGBoost Classifier

### 5. Model Evaluation
The trained models are evaluated using various evaluation metrics such as confusion matrix, accuracy score, precision, recall, and F1 score. Heatmaps are generated to visualize the confusion matrices.

### 6. Model Comparison and Selection
The performance of each model is compared based on the evaluation metrics. The model with the highest accuracy or the most suitable combination of precision and recall is selected as the best model for this specific problem.

### 7. Conclusion
The project concludes with a summary of the results obtained from different models. It also discusses potential future enhancements to improve the accuracy and reliability of the diabetes prediction system.

### Dataset Origin
The dataset used in this machine learning project is sourced from Kaggle, a popular platform for data science and machine learning enthusiasts. The dataset, titled "Diabetic Prediction Dataset," was selected due to its relevance to the project's aim of predicting the likelihood of diabetes based on various health-related features.

### Dataset Description
The "Diabetic Prediction Dataset" contains information about individuals, including various medical attributes that are potentially relevant to diabetes prediction. The dataset consists of a total of 768 instances, each with 9 features, and a binary target variable indicating whether an individual is diabetic (1) or not (0).

Features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

Target Variable:
- Outcome: 0 (Non-diabetic) or 1 (Diabetic)
- 
### Dataset Source
The dataset was sourced from the following Kaggle link: Diabetic Prediction Dataset. It is worth noting that the dataset has been preprocessed and cleaned for the purpose of this project.

### Purpose of Dataset
The primary purpose of utilizing this dataset is to develop and evaluate machine learning models capable of predicting the likelihood of an individual having diabetes based on their health attributes. The dataset provides a valuable opportunity to explore various classification algorithms and assess their performance in diagnosing diabetes.

By utilizing this dataset, we aim to contribute to the field of medical data analysis and predictive modeling, with potential applications in healthcare and early disease detection.

✏️**Dataset:** Diabetic Prediction Dataset. Retrieved from Kaggle

This documentation provides a comprehensive overview of the Diabetic Prediction Machine Learning project, including the objectives, methodology, implementation details, model evaluation, and conclusion.
