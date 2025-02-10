# PeterSide-HeartPredict_ML
## Peterside Hospital: Advanced Heart Disease Prediction Using Supervised Machine Learning Models

<img src="https://github.com/jamesehiabhi/PeterSide-HeartPredict_ML/blob/main/Displays/Heart_diease%20Cover.png" width="900" height="400"/> 

### INTRODUCTION
In the realm of healthcare, predictive analytics has emerged as a powerful tool to anticipate and mitigate risks associated with various diseases. Among these, heart disease remains a leading cause of mortality worldwide. Leveraging machine learning to predict heart disease can significantly enhance early diagnosis and treatment, potentially saving countless lives. This report delves into a comprehensive analysis of a heart disease dataset, employing a variety of machine learning models to predict the presence of heart disease. The insights derived from this analysis are not only pivotal for healthcare professionals but also for stakeholders and key opinion leaders in the healthcare industry.

### Table of Contents
- [Project Objective](#project-objective).
- [Data Sources](#data-sources).
- [Dataset Overview](#dataset-overview).
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda).
- [Machine Learning Models](#machine-learning-models).
- [Evaluation Metrics](#evaluation-metrics).
- [Key Findings and Insights](#key-findings-and-insights).
- [Recommendations](#recommendations).
- [Conclusion](#conclusion)


### 🫀Project Objective

The project aim to develop a machine-learning model for predicting the likelihood of a person having a heart disease based on there health features.

### 🫀Data Sources
The Dataset used in this project was provided by 10Alytics. The dataset contains relevant health data from patients at **Peterside Hospital**, including demographic information, medical history, lifestyle factors and results from diagnostic test.

### 🫀Dataset Overview
The dataset includes several features relevant to heart disease diagnosis:
- **Demographics:** Age, Sex.
- **Clinical Indicators:** Chest pain type, Resting blood pressure, Cholesterol levels, Fasting blood sugar, Electrocardiographic results.
- **Exercise-related Metrics:** Maximum heart rate achieved, Exercise-induced angina, ST depression.
- **Other Factors:** Number of major vessels observed via fluoroscopy, Thalassemia condition.
- **Target Variable:** Presence of heart disease (1 = Yes, 0 = No).
 

### 🫀Exploratory Data Analysis (EDA)
**Data Cleaning and Preprocessing**

Before diving into model building, it's crucial to ensure the dataset is clean and preprocessed. The dataset was inspected for missing values, and fortunately, none were found. The data types of the features were also verified to ensure they align with their descriptions.

**Statistical Summary**

A statistical summary of the dataset reveals key insights:
- The average **age of patients** is approximately 54 years.
- The majority of **patients** are male (68.3%).
- The average **resting blood pressure** is around 131.62 mm Hg.
- The average **cholesterol level** is 246.26 mg/dl.
- About 14.85% of patients have **fasting blood sugar** levels greater than 120 mg/dl.
- The **target variable** indicates that 54.46% of the patients have heart disease.

<img src="https://github.com/jamesehiabhi/PeterSide-HeartPredict_ML/blob/main/Displays/Stat.png" alt="Displays" width="800" height="300"/> 

**Data Visualization**
Visualizations were employed to better understand the distribution and relationships between features:

- **Age Distribution:** The age distribution is relatively normal, with most patients aged between 47 and 61 years.
- **Gender Distribution:** Males constitute a significant majority, which may influence the model's performance.
- **Chest Pain Types:** The dataset includes patients with various types of chest pain, with typical angina being the most common.
- **Cholesterol Levels:** Cholesterol levels are generally high, with a mean of 246.26 mg/dl.
- **Target Variable Distribution:** The dataset is slightly imbalanced, with more patients having heart disease.

<img src="https://github.com/jamesehiabhi/PeterSide-HeartPredict_ML/blob/main/Displays/EDA.png" alt="Displays" width="900" height="500"/> 

**Feature Engineering**

To enhance the predictive power of the models, feature engineering was performed:
- **Normalization:** Features were normalized to ensure they are on a similar scale, which is crucial for algorithms sensitive to the magnitude of data.
- **Feature Renaming:** For better readability and understanding, some columns were renamed to more descriptive names.

### 🫀Machine Learning Models
The heart disease model was built using a supervised machine-learning approach. Training and test data were split 80:20. Various machine learning models were employed to predict the presence of heart disease. The models included:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**
- **Gaussian Naive Bayes**
- **Stochastic Gradient Descent (SGD)**
- **Decision Tree**

After extensive experimentation and hyperparameter tuning, the final machine learning model was selected based on the performance and generalisation capabilities.

**Performance Evaluation**
The models were assessed using multiple metrics:
- **Accuracy:** Measures the overall correctness of predictions.
- **Precision & Recall:** Evaluate the model’s ability to correctly identify heart disease cases.
- **F1-score:** Balances precision and recall.
- **AUC-ROC:** Assesses the model’s discriminatory power.

### 🫀Evaluation Metrics
To access the performance of machine learning model, the following evaluation metrics were used:
- **Accuracy (86.89):** This indicates the proportion of correctly classified case of heart disease patient in the test data. In this case, 86.9% of the predictions were accurate.
- **Precision (85.29):** This measures the proportion of positive predictions that were actually correct (correctly identified cases of heart disease patient).
- **Recall (90.62):** This measures the proportion of actual heart disease cases that were correctly identified by the model.
- **AUC-ROC (86.69):** This metric reflects the model's ability to distinguish between healthy and heart disease cases.

**Confusion Matrix Analysis**
The **confusion matrix** highlighted that the model correctly classified most patients but had some misclassifications, however, the top-performing models (**Random Forest and Naïve Bayes**) were analyzed to understand the true positives, true negatives, false positives, and false negatives. Both models demonstrated a balanced performance, with a higher number of true positives and true negatives, indicating their robustness in predicting heart disease.

<img src="https://github.com/jamesehiabhi/PeterSide-HeartPredict_ML/blob/main/Displays/Scores.png" alt="Displays" width="1000" height="500"/>  

### 🫀Key Findings and Insights

**1. Predictors of Heart Disease**
Data analysis and model feature importance reveal the following significant predictors:
- **High Cholesterol (chol):** Individuals with cholesterol levels above 240 mg/dL were significantly more likely to have heart disease.
- **Resting Blood Pressure (trestbps):** Patients with systolic blood pressure over 140 mmHg showed higher risk.
- **Exercise-Induced Angina (exang):** Strong correlation with heart disease, indicating reduced exercise tolerance among at-risk patients.
- **Age and Gender:**
  - Risk increased significantly after age 50.
  - Males exhibited higher prevalence compared to females.

**2. Predictive Model Results**
The machine learning approach applied multiple algorithms, with Random Forest and Naïve Bayes providing the best balance of performance:

- **Accuracy: 86.9%** (indicating robust overall prediction).
- **Precision: 90%** (suggesting reliable identification of true positives).
- **Recall: 90.6** (ensuring minimal missed cases of heart disease).
- **AUC-ROC Score: 87.02%** reflecting excellent discrimination between positive and negative cases.

**3. Patient Risk Stratification**
The model categorized patients into risk levels, enabling personalized recommendations:

- **Low Risk:** Predominantly younger individuals with normal cholesterol and blood pressure levels.
- **Medium Risk:** Patients with moderate elevations in cholesterol or blood pressure.
- **High Risk:** Older patients or those with multiple elevated metrics, including angina or thalassemia conditions.

**4. Data Imbalance:** The slight imbalance in the target variable did not significantly impact the model performance, but future analyses could explore techniques like SMOTE to further balance the dataset.

### 🫀Recommendations
1.	**Feature Importance:**

  - Exercise-induced angina and ST depression levels were significant predictors.
  - Given the high predictive accuracy of the models, healthcare providers can leverage these models for early screening of heart disease, especially in high-risk populations.

2.	**Model Deployment:**

  - Random Forest provides an interpretable and effective approach for real-world applications.
  - The top-performing models (**Naïve Baye and Random Forest**) should be considered for deployment in clinical settings, with continuous monitoring and updates as more data becomes available.

3.	**Clinical and Educational Implementation:**
  - Integrating predictive models into hospital workflows can enhance early diagnosis.
  - Educating patients about the importance of monitoring key health metrics like cholesterol levels, blood pressure, and heart rate can aid in early detection and prevention.
  - Targeted campaigns like Designing educational materials emphasizing lifestyle changes to manage cholesterol, blood pressure and community screening. 

_**Implementing the model leads to enhanced early diagnosis rates, which reduces the burden on healthcare systems. It also results in significant cost savings, as preventative care is less expensive than late-stage treatments. Additionally, by empowering patients and providers with actionable data, the model can significantly improve health trajectories.**_

<img src="https://github.com/jamesehiabhi/PeterSide-HeartPredict_ML/blob/main/Displays/Cover%202.png" alt="Displays" width="800" height="400"/> 

________________________________________
### 🫀CONCLUSION
This project highlights the revolutionary impact of machine learning in the healthcare industry. Predicting heart disease risks with precision empowers clinicians to make informed decisions and fosters a proactive approach to health. Utilizing a rich dataset and advanced facilities at Peterside Hospital, the model identifies high-risk individuals early, paving the way for personalized care and timely interventions. This not only enhances patient outcomes but also strengthens **Peterside Hospital's** standing as a leader in healthcare innovation. With ongoing refinement, these models promise to further elevate the quality of medical diagnoses and patient care. 
By doing so, we can move closer to a future where heart disease is not just treatable but preventable.🩺⚕️❤️‍🩹
________________________________________

<br>

### *Kindly share your feedback and I am happy to Connect 🌟*

<img src="https://github.com/jamesehiabhi/PeterSide-HeartPredict_ML/blob/main/Displays/My%20Card1.jpg" alt="Displays" width="600" height="150"/>

