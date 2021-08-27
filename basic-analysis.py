import os

import pandas as pd  # for data analysis
import numpy as np  # for mathematical calculations
import seaborn as sns  # for data visualization

import matplotlib.pyplot as plt

"""
File to use pandas machine learning suite to perform a binary classification and predict whether a loan applicant will 
be approved or denied. Once the algorithm reaches a high degree of accuracy, it could be used as a substitute for 
human review on future applicants.

Data is contained in two CSV files, LoanPredictionTrainData.csv and LoanPredictionTestData.csv. The TrainData file 
contains the columns Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, 
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, and Loan_Status. The test data file 
contains the same columns, with the emission of the "Loan_status" column, and is used to assess the algorithm's 
accuracy. Both files were retrieved from AnalyticsVidhya.com."""


def main():
    # Read data
    train_data = pd.read_csv("LoanPredictionTrainData.csv")
    test_data = pd.read_csv("LoanPredictionTestData.csv")

    # Create copies to preserve original files if changed
    train_original = train_data.copy()
    test_original = test_data.copy()

    # print_data_overview(train_data)
    perform_univariate_analysis(train_data, 'TrainData')
    perform_bivariate_analysis(train_data, 'TrainData')

    # Reformat inconsistent data to allow deeper visualization and analysis
    train_data['Dependents'].replace('3+', 3, inplace=True)
    test_data['Dependents'].replace('3+', 3, inplace=True)
    train_data['Loan_Status'].replace('N', 0, inplace=True)
    train_data['Loan_Status'].replace('Y', 1, inplace=True)

    # Create a heatmap of correlations
    create_heatmap(train_data, 'TrainData')


def print_data_overview(data):
    """
    View the shape of a pandas DataFrame and it's columns.
    :param data: The pandas DataFrame to view.
    """
    print("DATA SHAPE\n" + str(data.shape) + "\n")
    print("COLUMN INFORMATION")
    print(data.dtypes)


def perform_univariate_analysis(data, data_name):
    """
    Create a series of plots and graphs showing univariate statistics for the data set.

    Reads data from the given file, and creates and saves png's containing histograms, frequency charts, and
    boxplots for the various columns.
    :param data: The DataFrame to chart.
    :param data_name: The prefix for the png to be saved with.
    """
    try:
        os.mkdir(data_name + 'UnivariateAnalysis')
    except FileExistsError:
        pass

    data_name = data_name + 'UnivariateAnalysis/' + data_name
    # Create plots of loan acceptance frequency.
    data['Loan_Status'].value_counts(normalize=True).plot.bar(title='Loan Status')
    plt.savefig(data_name + 'LoanStatusFrequencyChart')

    # Create image of categorical variables
    plt.figure(1)
    plt.subplot(221)
    data['Gender'].value_counts(normalize=True).plot.bar(figsize=(20, 10), title='Gender')
    plt.subplot(222)
    data['Married'].value_counts(normalize=True).plot.bar(title='Married')
    plt.subplot(223)
    data['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self Employed')
    plt.subplot(224)
    data['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit History')
    plt.savefig(data_name + 'CategoricalVariableFrequency')

    # Create plots of ordinal variables
    plt.figure(1)
    plt.subplot(131)
    data['Dependents'].value_counts(normalize=True).plot.bar(figsize=(18, 10), title='Dependents')
    plt.subplot(132)
    data['Education'].value_counts(normalize=True).plot.bar(title='Education')
    plt.subplot(133)
    data['Property_Area'].value_counts(normalize=True).plot.bar(title='Property Area')
    plt.savefig(data_name + 'OrdinalVariableFrequency')

    # Create plots of numerical variables
    plt.figure(1)
    plt.subplot(121)
    sns.histplot(data['ApplicantIncome'], stat='proportion', kde=True)
    plt.subplot(122)
    data['ApplicantIncome'].plot.box(figsize=(16, 5))
    plt.savefig(data_name + 'ApplicantIncomeFrequency')
    plt.figure(1)
    data.boxplot(column='ApplicantIncome', by='Education')
    plt.suptitle("")
    plt.savefig(data_name + 'EducationOnApplicantIncome')

    plt.figure(2)
    plt.subplot(121)
    sns.histplot(data['CoapplicantIncome'], stat='proportion', kde=True)
    plt.subplot(122)
    data['CoapplicantIncome'].plot.box(figsize=(16, 5))
    plt.savefig(data_name + 'CoapplicantIncomeFrequency')
    plt.close('All')


def perform_bivariate_analysis(data, data_name):
    """
    Create a series of plots and graphs showing bivariate statistics for the data set.

    Reads data from the given file, and creates and saves png's containing histograms, frequency charts, and
    boxplots for the various columns.
    :param data: The DataFrame to chart.
    :param data_name: The prefix for the png to be saved with.
    """
    try:
        os.mkdir(data_name + 'BivariateAnalysis')
    except FileExistsError:
        pass
    data_name = data_name + 'BivariateAnalysis/' + data_name

    # Create charts comparing the categorical variables to loan status.
    gender = pd.crosstab(data['Gender'], data['Loan_Status'])
    married = pd.crosstab(data['Married'], data['Loan_Status'])
    self_employed = pd.crosstab(data['Self_Employed'], data['Loan_Status'])
    dependents = pd.crosstab(data['Dependents'], data['Loan_Status'])
    education = pd.crosstab(data['Education'], data['Loan_Status'])

    df_list = [married, self_employed, education, gender]
    fig, axes = plt.subplots(2, 2)
    count = 0
    for r in range(2):
        for c in range(2):
            df_list[count].div(df_list[count].sum(1).astype(float), axis=0) \
                .plot(kind='bar', stacked=True, ax=axes[r, c], figsize=(10, 10))
            count += 1
    plt.savefig(data_name + 'CategoricalVariablesOnLoanStatus')

    dependents.div(dependents.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.savefig(data_name + 'DependentsOnLoanStatus')
    plt.close()

    # Create charts analyzing various income factors to loan status.
    fig, axes = plt.subplots(2, 2)
    plt.subplot(221)
    # Analyze based on mean income.
    # Notice: Mean income appears to have no impact on loan status. Possibly obscured by household income, related to
    # marriage frequency at higher income.
    data.groupby('Loan_Status')['ApplicantIncome'].mean().plot(kind='bar', xlabel='Loan Status', ylabel='Mean Income',
                                                               figsize=(11, 11), ax=axes[0, 0])

    # Split mean income into several categories for visual clarity
    plt.subplot(222)
    bins = [0, 2500, 4000, 6000, 81000]
    group = ['Low', 'Average', 'High', 'Very High']
    data['Income_Bin'] = pd.cut(data['ApplicantIncome'], bins, labels=group)
    income_bin = pd.crosstab(data['Income_Bin'], data['Loan_Status'])
    # Notice: Income still appears independent of loan status when grouped.
    income_bin.div(income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, xlabel='Applicant Income',
                                                                 ylabel='Proportion', ax=axes[0, 1])
    # Analyze coapplicant income compared to loan loan approval.
    bins = [0, 1000, 3000, 42000]
    group = ['Low', 'Average', 'High']
    data['Coapplicant_Income_Bin'] = pd.cut(data['CoapplicantIncome'], bins, labels=group)
    coapplicant_income_bin = pd.crosstab(data['Coapplicant_Income_Bin'], data['Loan_Status'])
    # Notice: Higher coapplicant income correlates with lower loan approval. This is counter-intuitive, and
    # the next test should likely be household income.
    coapplicant_income_bin.div(coapplicant_income_bin.sum(1).astype(float), axis=0)\
        .plot(kind='bar', stacked=True, xlabel='Coapplicant Income', ylabel='Proportion', ax=axes[1, 0])
    # Analyze combined income compared to loan approval.
    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    bins = [0, 2500, 4000, 6000, 81000]
    group = ['Low', 'Average', 'High', 'Very High']
    data['Total_Income_Bin'] = pd.cut(data['Total_Income'], bins, labels=group)
    total_income_bin = pd.crosstab(data['Total_Income_Bin'], data['Loan_Status'])
    # Notice: Lower household income correlates with lower loan approval. This explains the prior data.
    total_income_bin.div(total_income_bin.sum(1).astype(float), axis=0)\
        .plot(kind='bar', stacked=True, xlabel='Household Income', ylabel='Proportion', ax=axes[1, 1])
    plt.savefig(data_name + 'IncomeOnLoanStatus')
    plt.close()

    # Create chart comparing loan amount to loan approval
    bins = [0, 100, 200, 700]
    group = ['Low', 'Average', 'High']
    data['LoanAmount_Bin'] = pd.cut(data['LoanAmount'], bins, labels=group)
    loan_amount_bin = pd.crosstab(data['LoanAmount_Bin'], data['Loan_Status'])
    loan_amount_bin.div(loan_amount_bin.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(6, 8),
                                                                           xlabel='Loan Amount', ylabel='Proportion')
    plt.savefig(data_name + 'LoanAmountOnLoanStatus')
    plt.close()

    data = data.drop(['Income_Bin', 'Coapplicant_Income_Bin', 'LoanAmount_Bin',
                      'Total_Income_Bin', 'Total_Income'], axis=1)


def create_heatmap(data, data_name):
    """
    Create a Seaborn heatmap to visualize the correlation of all variables.
    :param data: The data to visualize.
    :param data_name: The prefix to save the heatmap with.
    """
    matrix = data.corr()
    f, axes = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, vmax=0.8, square=True, cmap="BuPu")
    plt.savefig(data_name + 'Heatmap')


if __name__ == '__main__':
    main()
