# Credit Risk Project: An Overview
* I analysed a credit risk dataset to find patterns which show us the more risky customers.
* After the analysis I used Logistic Regression (~90% accuracy), Random Forest and GradienBoosting Classifier to predict the loan status.
## Code
* Python 3 (ipykernel)
* Packages: pandas, numpy, sklearn, matplotlib, seaborn, plotly
* Dataset source: (https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

## Dataset
Columns of the dataset:
* person_age: Age of the individual applying for the loan.
* person_income: Annual income of the individual.
* person_home_ownership: Type of home ownership of the individual.
* person_emp_length: Employment length of the individual in years.
* loan_intent: The intent behind the loan application.
* loan_grade: The grade assigned to the loan based on the creditworthiness of the borrower.
* loan_amnt: The loan amount requested by the individual.
* loan_int_rate: The interest rate associated with the loan.
* loan_status: Loan status, where 0 indicates non-default and 1 indicates default.
* loan_percent_income: The percentage of income represented by the loan amount.
* cb_person_default_on_file: Historical default of the individual as per credit bureau records.
* cb_preson_cred_hist_length: The length of credit history for the individual.

## Analysis
### First steps
First I get some general informations about the dataset using .info(); .describe(); .corr() and .hist() functions.
I droped the rows with null values.
I looked for outliers with boxplot, and removed them.
I used scatter plots to see the correlations.


* People's credit history begins at age of 17-26
![alt text](https://github.com/trauerj/Credit_Risk_Project/blob/main/Images/hist_length_age_plot.png)
* People with default on file have a higher loan interest rate in general
![alt text](https://github.com/trauerj/Credit_Risk_Project/blob/main/Images/default_on_file_int_rate_plot.png)
### Grouping numerical data for further analysis
I used pd.cut function for grouping. I plotted the groups "distributions" with pie charts.
 * Most of the customers (50.8%) live in a rented property or have a mortgage on the property what they live in (41.2%). Just the 7.7% of the customers own their home.
 * Most of the customers' age is between 20 and 39 (95.5%). (It is a good explonation for the dsitribution of the ownerships.)
 * Most common loan amounts are between 0 and 10,000 (65.6%).
 * Most customers' income is in the range of 25,000 - 50,000 (35.6%). The smallest income group is <25,000 (~7%).
 * Most of the customers' annual income doesn't reach the 20% of the loan amount (69.4%).
 * Despite of that, more than the half of the loans are graded A or B (64.8%).
### Annalysis with pivot tables, cross tabulations, heatmaps and stacked charts
I analysed the correlations betwwen the created groups and the loan status to find patterns.
- #### Pivot Table 1:
 - Customer with a default on file means a higher risk.
 ![alt text](https://github.com/trauerj/Credit_Risk_Project/blob/main/Images/default_on_file_status_plot.png)
- #### Pivot Table 4:
 - People's loan status more likely be default with lower income. (0-25K (51.4%); 25-50K (~28%))
 - As the income grows the chance for a default loan status decrease.
 ![alt text](https://github.com/trauerj/Credit_Risk_Project/blob/main/Images/income_status_plot.png)
- #### Pivot Table 5:
 - Interestingly, people with higher income/loan amount ratio, more likely have a default loan status.
 - (Note: There are very few customer with a higher (>40%) income/loan amount ratio, so the representativeness of this tendency/data is not the best (Not representative).)
- #### Pivot Table 6:
 - A higher loan amount means a higher chance for a default loan status. (10-15K (23%); 15-35K (~32.6%)).
![alt text](https://github.com/trauerj/Credit_Risk_Project/blob/main/Images/amount_group_status_plot.png)

## Model Building
First, I transformed the categorical variables into dummy variables.
I also split the data into train and test sets with a test size 20%.

I tried three different models:
* Logistic Regression - Baseline for the model
* Random Forest Classifier
* Gradient Boosting Classifier

|      Model      | Right predictions | Wrong predictions | Accuracy |
|----------------:|------|------|------|
|Linear Regression| 5153 | 573 | ~90% |
|Random Forest    | 4624 | 1102 | ~81% |
|Gradient Boosting| 4602 | 1124 | ~81% |
