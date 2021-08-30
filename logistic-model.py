import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Read data
    train_data = pd.read_csv("LoanPredictionTrainData.csv")
    test_data = pd.read_csv("LoanPredictionTestData.csv")

    # Create copies to preserve original files if changed
    train_original = train_data.copy()
    test_original = test_data.copy()

    # Align data format to ease statistical analysis.
    correct_missing_values(train_data)
    correct_missing_values(test_data)
    # Treat outliers by adding a log-column to LoanAmount
    train_data['LoanAmount_Log'] = np.log(train_data['LoanAmount'])
    test_data['LoanAmount_Log'] = np.log(test_data['LoanAmount'])
    # Drop irrelevant Loan_ID variable
    train_data.drop('Loan_ID', inplace=True, axis=1)
    test_data.drop('Loan_ID', inplace=True, axis=1)

    # Create logistic regression model.
    log_reg_model = create_logistic_model(train_data, 'Loan_Status')

    # Create another model with additional features to compare accuracy
    train_data = create_additional_features(train_data)
    test_data = create_additional_features(test_data)

    # Test logistic model with additional engineered features
    log_reg_model = create_logistic_model(train_data, 'Loan_Status')



    # Reapply the Loan_ID column and output the results to a csv file.
    # predicted_data['Loan_ID'] = test_original['Loan_ID']
    # pd.DataFrame(predicted_data, columns=['Loan_ID', 'Loan_Status']).to_csv(path_or_buf='PredictedApproval', index=False)


def apply_model(model, data, class_column_name='class', true_label='1', false_label='0'):
    """
    Apply the given model onto a data set and add the predicted classes to it.
    :param model: The model to apply.
    :param data: The data to classify.
    :param class_column_name: The desired name of the classification column in the output.
    :param true_label: The desired name of the 'positive' class in the output.
    :param false_label: The desired name of the 'negative' class in the output.
    """
    data = pd.get_dummies(data)
    pred_classes = model.predict(data)
    data[class_column_name] = pred_classes
    data[class_column_name].replace(0, false_label, inplace=True)
    data[class_column_name].replace(1, true_label, inplace=True)
    return data


def correct_missing_values(data):
    """
    Correct the missing values in the data. Function is custom-made for the LoanPredictionTestData and
    LoanPredictionTrainData sets and will fail if used with differently formatted data.
    :param data: The files LoanPredictionTrainData.csv or LoanPredictionTestData.csv.
    :return: The file with it's missing data corrected to an appropriate value.
    """
    # View what data is currently missing
    # print(data.isnull().sum())

    # Fill missing categorical data with the most common values
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    # Loan_Amount_Term isn't technically categorical, but it should be considered one due to the few possible values.
    print(data['Loan_Amount_Term'].value_counts())
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)

    # Fill missing Loan_Amount cells using median.
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)

    # Confirm no further data is missing
    # print(data.isnull().sum())


def create_additional_features(data):
    """
    Add the log total income, EMI, and balance income fields to the data. These fields are likely to affect the
    target variable.

    The total income column represents the combined Applicant Income and Coapplicant Income, and will likely correlate
    with higher approval rates. Because this data is skewed right, the log total income is used instead. The EMI
    (Equated Monthly Installment) is the monthly amount to be paid by the applicant, and will likely correlate with
    higher difficult paying the loan and therefore lower approval rates. The Balance Income is the income left after
    the EMI has been paid, and will likely correlate with higher approval rates.

    :param data: The data to add the columns to.
    :return: The data with the columns added.
    """
    # Create field for total income.
    # Notice: Data a right-skew.
    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']

    # Create field for log total income to correct right skew
    data['Total_Income_Log'] = np.log(data['Total_Income'])

    # Create field for EMI
    data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']

    # Create field for Balance Income
    # Multiply by 1000 to correct units
    data['Balance_Income'] = data['Total_Income'] - data['EMI'] * 1000

    # Drop variables used to create features to reduce noise
    data = data.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

    # Return altered data
    return data


def create_logistic_model(data, target_variable):
    """
    Create a logistic regression model to predict classification on the dataset.

    Performs the necessary conversions on the formatted data set, then utilizes the scikit-learn package to create and
    return a logistic model to predict the target variable.
    :param data: The data set, with irrelevant data removed, outliers treated, and null values filled.
    :param target_variable: The column name of the target variable.
    :return: A LogisticRegression designed to classify matching dummy data and predict the target variable.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Split target variable into second data set for sklearn.
    x = data.drop(target_variable, axis=1)
    y = data[target_variable]

    # Create dummies to make categorical comparisons simpler.
    x = pd.get_dummies(x)

    # Randomly split data into train and test data at a 7:3 proportion
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.3)

    # Create model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    LogisticRegression(C=1, max_iter=100000, n_jobs=1, penalty='l2', solver='liblinear', tol=0.0001,
                       verbose=0, warm_start=False)

    # Test the model's accuracy
    pred_cv = model.predict(x_cv)
    print('Accuracy of logistic regression model: ' + str(accuracy_score(y_cv, pred_cv)))

    return model


def create_decision_tree(data, target_variable):
    pass


if __name__ == '__main__':
    main()
