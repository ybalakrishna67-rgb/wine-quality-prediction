Wine Quality Prediction using Machine Learning

Project Overview

This project predicts the quality of red wine using different Machine Learning algorithms. The dataset contains several chemical properties of wine such as acidity, sugar, pH, sulphates, and alcohol. Based on these features, the model predicts whether the wine quality is good or not.

Dataset

The dataset used in this project is the Red Wine Quality Dataset from the UCI Machine Learning Repository.

Dataset file:

- winequality-red.csv

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Machine Learning Algorithms Implemented

The following algorithms were implemented in this project:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)

Project Steps

1. Import required libraries
2. Load the dataset
3. Data preprocessing and cleaning
4. Convert wine quality into binary classification
5. Split dataset into training and testing sets
6. Train multiple machine learning models
7. Evaluate models using accuracy score

Model Accuracy Results

Algorithm| Accuracy
Logistic Regression| 74.06%
Decision Tree| 72.18%
Random Forest| 79.06%
KNN| 61.25%

Best Model

The Random Forest Classifier achieved the highest accuracy of 79.06%, making it the best performing model for this dataset.

Random Forest performs better because it combines multiple decision trees and reduces overfitting, resulting in better prediction performance.

Conclusion

Machine learning algorithms can successfully predict wine quality based on chemical properties. Among all tested models, Random Forest provided the best accuracy for this dataset.
