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
from sklearn import metrics

sl = pd.read_excel('Preprocessed IBM Data.xlsm')
data = sl.copy()


'''MASK Employee Number'''
data['EmployeeNumber'] = np.arange(len(data))

inputs_train, inputs_test, targets_train, targets_test = train_test_split(data.drop('Attrition', axis=1),
                                                                          data['Attrition'], test_size=0.2,
                                                                          random_state=42)
'''Export to CSV'''
# print(inputs_train)
inputs_train.to_csv('inputs_train')
targets_train.to_csv('targets_train')
inputs_test.to_csv('inputs_test')
targets_test.to_csv('targets_test')

print(inputs_train.shape)
print(targets_train.shape)
print(inputs_test.shape)
print(targets_test.shape)
