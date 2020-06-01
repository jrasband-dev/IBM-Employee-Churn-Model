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

'''SQL SERVER CONNECTION'''
conn = pyodbc.connect(r'DRIVER={ODBC Driver 13 for SQL Server};' +
                      ('SERVER={server};' +
                       '{port};' +
                       'DATABASE={database};' +
                       'Trusted_Connection={Trusted_Connection};').format(server=''
                                                                        , port = 1443
                                                                        , database = ''
                                                                        , Trusted_Connection = 'yes'))
directory = 'C:\\'
sql = '''


SELECT 
	[EmployeeNumber]
	--,Attrition
	,CASE WHEN Attrition = 'Yes' THEN 0 ELSE 1									END AS 'Attrition'
	--,AGE
	,CASE WHEN Age BETWEEN 18 AND 30 THEN 1 ELSE 0								END AS 'AGE 18-30'
	,CASE WHEN Age BETWEEN 31 AND 40 THEN 1 ELSE 0								END AS 'AGE 31-60'

    --,BusinessTravel
	,Case WHEN BusinessTravel = 'Travel_Frequently' THEN 1 ELSE 0				END as 'Freq_Bus_Travel'
	,Case WHEN BusinessTravel = 'Travel_Rarely' THEN 1 ELSE 0					END as 'Rare_Bus_Travel'
	,Case WHEN BusinessTravel = 'Non-Travel' THEN 1 ELSE 0						END as 'Non-Travel'
	--,[DistanceFromHome]
	,CASE WHEN DistanceFromHome BETWEEN 0 AND 5 THEN 1 ELSE 0					END as 'Commute Distance 0-5mi'
	,CASE WHEN DistanceFromHome BETWEEN 6 AND 7 THEN 1 ELSE 0					END as 'Commute Distance 6-10mi'
	,CASE WHEN DistanceFromHome >= 11  THEN 1 ELSE 0					            END AS 'Commute Distance >=11mi'
    --,[EnvironmentSatisfaction]
	,Case WHEN EnvironmentSatisfaction = '1' THEN 1 ELSE 0						END as 'EnvironmentSatisfaction Survey Response 1'
	,Case WHEN EnvironmentSatisfaction = '2' THEN 1 ELSE 0						END as 'EnvironmentSatisfaction Survey Response 2'
	,Case WHEN EnvironmentSatisfaction = '3' THEN 1 ELSE 0						END as 'EnvironmentSatisfaction Survey Response 3'
	,Case WHEN EnvironmentSatisfaction = '4' THEN 1 ELSE 0						END as 'EnvironmentSatisfaction Survey Response 4'
    --,[JobSatisfaction]
	,Case WHEN JobSatisfaction = '1' THEN 1 ELSE 0								END as 'Job Satisfaction Survey Response 1'
	,Case WHEN JobSatisfaction = '2' THEN 1 ELSE 0								END as 'Job Satisfaction Survey Response 2'
	,Case WHEN JobSatisfaction = '3' THEN 1 ELSE 0								END as 'Job Satisfaction Survey Response 3'
	,Case WHEN JobSatisfaction = '4' THEN 1 ELSE 0								END as 'Job Satisfaction Survey Response 4'

    --,[MaritalStatus]
	,Case WHEN MaritalStatus = 'Single' THEN 1 ELSE 0							END as 'Single'
	,Case WHEN MaritalStatus = 'Married' THEN 1 ELSE 0							END as 'Married'
	,Case WHEN MaritalStatus = 'Divorced' THEN 1 ELSE 0							END as 'Divorced'
    --,[NumCompaniesWorked]
	,CASE WHEN NumCompaniesWorked Between 0 AND 4 THEN 1 ELSE 0					END as 'NumCompaniesWorkedAt 0-4'
	,CASE WHEN NumCompaniesWorked Between 5 AND 6 THEN 1 ELSE 0					END as 'NumCompaniesWorkedAt 5-6'
	,CASE WHEN NumCompaniesWorked Between 7 AND 8 THEN 1 ELSE 0					END as 'NumCompaniesWorkedAt 7-8'
	,CASE WHEN NumCompaniesWorked >= 9 THEN 1 ELSE 0							END as 'NumCompaniesWorkedAt >=9'
    --,[OverTime]
	,Case WHEN OverTime= 'Yes' THEN 1 ELSE 0									END as 'Overtime'
    --,YearsAtCompany
	,CASE WHEN YearsAtCompany Between 0 AND 5 THEN 1 ELSE 0					    END as 'Years At Company 0-5'
	,CASE WHEN YearsAtCompany Between 6 AND 15 THEN 1 ELSE 0					END as 'Years At Company 6-15'
	,CASE WHEN YearsAtCompany Between 16 AND 30 THEN 1 ELSE 0				    END as 'Years At Company 16-30'
	,CASE WHEN YearsAtCompany >= 31 THEN 1 ELSE 0				                END as 'Years At Company >=31'
    --,YearsInCurrentRole
	,CASE WHEN YearsInCurrentRole Between 0 AND 6 THEN 1 ELSE 0					END as 'Years In Current Role 0-6'
	,CASE WHEN YearsInCurrentRole Between 7 AND 12 THEN 1 ELSE 0				END as 'Years In Current Role 7-12'
	,CASE WHEN YearsInCurrentRole >=13 THEN 1 ELSE 0				            END as 'Years In Current Role >=13'
	
    --,YearsWithCurrManager
	,CASE WHEN MontlyIncome BETWEEN 0 and 2000 THEN 1 ELSE 0					END as 'Monthly Income 0-4000'
	,CASE WHEN MontlyIncome BETWEEN 4001 and 6000 THEN 1 ELSE 0					END as 'Monthly Income 4001-6000'
	,CASE WHEN MontlyIncome BETWEEN 6001 and 8000 THEN 1 ELSE 0					END as 'Monthly Income 6001-8000'
	,CASE WHEN MontlyIncome  >= 8001 THEN 1 ELSE 0				                END as 'Monthly Income >=8001'

FROM [TheCrew].[dbo].[IBM]


'''

sl = pd.read_sql(sql, conn)
# print(sl.info())
data = sl.copy()
conn.close()

'''MASK LOAN AND BORROWER ID'''
data['EmployeeNumber'] = np.arange(len(data))

# data.to_excel(directory + '/'+'LoanData.xlsx')

inputs_train, inputs_test, targets_train, targets_test = train_test_split(data.drop('Attrition', axis=1),
                                                                          data['Attrition'], test_size=0.2,
                                                                          random_state=42)

'''Export to CSV'''
# print(inputs_train)
inputs_train.to_csv(directory + '/' + 'inputs_train')
targets_train.to_csv(directory + '/' + 'targets_train')
inputs_test.to_csv(directory + '/' + 'inputs_test')
targets_test.to_csv(directory + '/' + 'targets_test')

print(inputs_train.shape)
print(targets_train.shape)
print(inputs_test.shape)
print(targets_test.shape)
