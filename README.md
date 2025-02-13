
# House Price Prediction using Machine Learning

This project demonstrates the use of machine learning to predict house prices based on various features like the applicant's income, credit history, and more. The model used for prediction is a RandomForestClassifier, which is trained on historical data and then used to predict house prices on unseen data.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data](#data)
4. [Model Training](#model-training)
5. [Prediction and Results](#prediction-and-results)
6. [Evaluation](#evaluation)
7. [License](#license)

## Project Overview

The goal of this project is to predict whether a loan will be approved based on various applicant features. The model uses a Random Forest Classifier algorithm to predict if a loan is approved or not.

### Key Features:
- **Loan_ID**: Unique identifier for each loan
- **Gender**: Gender of the applicant
- **Married**: Whether the applicant is married
- **Dependents**: Number of dependents
- **Education**: Education level of the applicant
- **Self_Employed**: Whether the applicant is self-employed
- **Property_Area**: The area in which the property is located (Urban, Semiurban, or Rural)
- **ApplicantIncome**: Applicant’s monthly income
- **CoapplicantIncome**: Coapplicant’s monthly income
- **LoanAmount**: Loan amount requested by the applicant
- **Loan_Amount_Term**: Term of the loan in months
- **Credit_History**: Credit history of the applicant (1 for good, 0 for poor)

The prediction aims to classify if the loan will be approved (1) or rejected (0).

## Installation

To get started with this project, you'll need Python and several dependencies installed. Follow the steps below to set up your environment:

1. Clone this repository:
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data

The dataset used for this project is a typical loan application dataset. It includes features like applicant income, credit history, and more. You can find the dataset in the `data/` directory, or you can use your own dataset as needed.

## Model Training

In this project, the model is trained using a **RandomForestClassifier**. The key steps include:

1. **Loading Data**: The training data is loaded into a DataFrame for further preprocessing.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale continuous features.
3. **Model Training**: Train a RandomForestClassifier with the following parameters:
   - `n_estimators = 200`
   - `max_features = 'sqrt'`
   - `criterion = 'gini'`
   - `random_state = 4`
   - `n_jobs = None`

4. **Model Fitting**: The model is fit to the training data, and its accuracy is evaluated using cross-validation.

## Prediction and Results

Once the model is trained, we use it to predict house prices on the test data:

```python
# Prediction on test_set
final_model = one_model(RandomForestClassifier(random_state = 4, n_jobs = None, n_estimators = 200, max_features = 'sqrt', criterion = 'gini'), model=True)

predictions = final_model.predict(df_test)

# Add predictions to the test DataFrame
df_test['Prediction'] = predictions
df_test.head()
```

This will add a new `Prediction` column to the test dataset, where the model predicts whether the loan is approved or rejected (1 or 0).

## Evaluation

After making predictions, you can evaluate the model using performance metrics such as accuracy, confusion matrix, and ROC curve. You can also explore feature importance to understand which features are driving the model's decisions.

Example:
```python
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
```

## License

This project is licensed under the Apache 2.0
