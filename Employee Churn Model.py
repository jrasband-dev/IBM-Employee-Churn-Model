import openpyxl as opx
import os
import shutil as s
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import datetime as dt
import pyodbc
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import linear_model
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit

'''SET UP'''
sns.set()
# pd.options.display.max_columns = None
pd.options.display.max_rows = None

'''OPEN THE FILES'''
inputs_train = pd.read_csv('inputs_train', index_col=0)
targets_train = pd.read_csv('targets_train',index_col=0, header=None)
inputs_test = pd.read_csv('inputs_test', index_col=0)
targets_test = pd.read_csv('targets_test', index_col=0)

targets_test['EmployeeNumber'] = inputs_test['EmployeeNumber']

'''STATIC DEFINITIONS'''

ref_categories = ['AGE 31-60'
                    ,'Freq_Bus_Travel'
                    ,'Commute Distance >=10mi'
                    ,'EnvironmentSatisfaction Survey Response 1'
                    ,'Job Satisfaction Survey Response 1'
                    ,'Single'
                    ,'NumCompaniesWorkedAt >=3'
                    , 'Total Working Years 8-11'
                    ,'Monthly Income 0-4000']


targets_train2 = targets_train.copy()
targets_train2['EmployeeNumber'] = inputs_train['EmployeeNumber']
'''REMOVE REFERENCE CATEGORIES'''
inputs_train_w_ref_cat = inputs_train.loc[:, ['AGE 18-30'
                                                ,'AGE 31-60'
                                                ,'Total Working Years 0-7'
                                                ,'Total Working Years 8-11'
                                                ,'Total Working Years >=12'
                                                ,'Freq_Bus_Travel'
                                                ,'Rare_Bus_Travel'
                                                ,'Non-Travel'
                                                ,'Commute Distance 0-3mi'
                                                ,'Commute Distance 3-9mi'
                                                ,'Commute Distance >=10mi'
                                                ,'EnvironmentSatisfaction Survey Response 1'
                                                ,'EnvironmentSatisfaction Survey Response 2'
                                                ,'EnvironmentSatisfaction Survey Response 3'
                                                ,'EnvironmentSatisfaction Survey Response 4'
                                                ,'Job Satisfaction Survey Response 1'
                                                ,'Job Satisfaction Survey Response 2'
                                                ,'Job Satisfaction Survey Response 3'
                                                ,'Job Satisfaction Survey Response 4'
                                                ,'Single'
                                                ,'Married'
                                                ,'Divorced'
                                                ,'Is Male'
                                                ,'NumCompaniesWorkedAt 0-2'
                                                ,'NumCompaniesWorkedAt >=3'
                                                ,'Overtime'
                                                ,'Monthly Income 0-4000'
                                                ,'Monthly Income 4001-6000'
                                                ,'Monthly Income 6001-8000'
                                                ,'Monthly Income >=8001'
                                              ]]

inputs_train = inputs_train_w_ref_cat.drop(ref_categories, axis=1)
'''RUN REGRESSION AND PLACE INTO A SUMMARY TABLE'''
reg = LogisticRegression()
reg.fit(inputs_train, targets_train)
# print(reg.intercept_)
# print(reg.coef_)
feature_name = inputs_train.columns.values
summary_table = pd.DataFrame(columns=['feature_name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
print(summary_table)

'''BUILD LOGISTICS MODEL'''


class LogisticRegression_with_p_values:
    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs, max_iter=100000)

    def fit(self, X, y):
        self.model.fit(X, y)
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        f_ij = np.dot((X / denom).T, X)
        Cramer_Rao = np.linalg.inv(f_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [scipy.stats.norm.sf(abs(x)) * 2 for x in z_scores]
        self.p_values = p_values


reg2 = LogisticRegression_with_p_values()
reg2.fit(inputs_train, targets_train)
feature_name = inputs_train.columns.values
summary_table = pd.DataFrame(columns=['feature_name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg2.p_values
# print(p_values)
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
print(summary_table)
summary_table.to_excel(directory + '/' + 'ModelSummary.xlsx')

# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.scatter(inputs_train, targets_train, color='black', zorder=20)
# X_test = np.linspace(-5, 10, 300)
# loss = expit(X_test * reg.coef_ + reg.intercept_).ravel()
# plt.plot(X_test, loss, color='red', linewidth=3)
# plt.plot(X_test, reg.coef_ * X_test + reg.intercept_, linewidth=1)
# plt.axhline(.5, color='.5')
#
# plt.ylabel('y')
# plt.xlabel('X')
# plt.xticks(range(-5, 10))
# plt.yticks([0, 0.5, 1])
# plt.ylim(-.25, 1.25)
# plt.xlim(-4, 10)
# plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
#            loc="lower right", fontsize='small')
# plt.tight_layout()
# plt.show()

'''VALIDATION'''
inputs_test_w_ref_cat = inputs_test.loc[:, ['AGE 18-30'
                                                ,'AGE 31-60'
                                                ,'Total Working Years 0-7'
                                                ,'Total Working Years 8-11'
                                                ,'Total Working Years >=12'
                                                ,'Freq_Bus_Travel'
                                                ,'Rare_Bus_Travel'
                                                ,'Non-Travel'
                                                ,'Commute Distance 0-3mi'
                                                ,'Commute Distance 3-9mi'
                                                ,'Commute Distance >=10mi'
                                                ,'EnvironmentSatisfaction Survey Response 1'
                                                ,'EnvironmentSatisfaction Survey Response 2'
                                                ,'EnvironmentSatisfaction Survey Response 3'
                                                ,'EnvironmentSatisfaction Survey Response 4'
                                                ,'Job Satisfaction Survey Response 1'
                                                ,'Job Satisfaction Survey Response 2'
                                                ,'Job Satisfaction Survey Response 3'
                                                ,'Job Satisfaction Survey Response 4'
                                                ,'Single'
                                                ,'Married'
                                                ,'Divorced'
                                                ,'Is Male'
                                                ,'NumCompaniesWorkedAt 0-2'
                                                ,'NumCompaniesWorkedAt >=3'
                                                ,'Overtime'
                                                ,'Monthly Income 0-4000'
                                                ,'Monthly Income 4001-6000'
                                                ,'Monthly Income 6001-8000'
                                                ,'Monthly Income >=8001']]

# print(inputs_test_w_ref_cat)

inputs_test = inputs_test_w_ref_cat.drop(ref_categories, axis=1)
inputs_train = inputs_test_w_ref_cat.drop(ref_categories, axis=1)
print(inputs_test.info())

'''TEST DATASET PROBABILITY OF DEFAULT'''
y_hat_test = reg2.model.predict(inputs_test)
# print(y_hat_test)
y_hat_test_proba = reg2.model.predict_proba(inputs_test)
# print(y_hat_test_proba)
y_hat_test_proba = y_hat_test_proba[:][:, 1] 
targets_test_temp = targets_test
targets_test_temp.reset_index(drop=True, inplace=True)
actual_predicted_probs = pd.concat([targets_test_temp, pd.DataFrame(y_hat_test_proba)], axis=1)
actual_predicted_probs.columns = ['Attrition', 'EmployeeNumber', 'y_hat_test_proba']
print(actual_predicted_probs)

'''ACCURACY OF TEST PROBABILITY'''
tr = 0.80
actual_predicted_probs['y_hat_test'] = np.where(actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)
print(pd.crosstab(actual_predicted_probs['Attrition'], actual_predicted_probs['y_hat_test'], rownames=['Actual'],
                  colnames=['Predicted']))
# print(pd.crosstab(actual_predicted_probs['GoodLoan'], actual_predicted_probs['y_hat_test'], rownames=['Actual'], colnames=['Predicted'])/actual_predicted_probs.shape[0])
print((pd.crosstab(actual_predicted_probs['Attrition'], actual_predicted_probs['y_hat_test'], rownames=['Actual'],
                   colnames=['Predicted']) / actual_predicted_probs.shape[0]).iloc[0, 0]
      + (pd.crosstab(actual_predicted_probs['Attrition'], actual_predicted_probs['y_hat_test'], rownames=['Actual'],
                     colnames=['Predicted']) / actual_predicted_probs.shape[0]).iloc[1, 1])
