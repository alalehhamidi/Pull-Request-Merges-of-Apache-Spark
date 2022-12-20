import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

raw_data = pd.read_csv('D:\Alaleh\FinalSparkData.csv')
q = raw_data['Size(LOC)'].quantile(0.95)
data = raw_data[raw_data['Size(LOC)']<q]
#print(data.head())
#print(data.describe())

x = data[['Size(LOC)', 'NoFiles', 'NoCommits', 'Experience', 'NoComments', 'NoCommentingDevs', 'NoAuthorComments',
          'NoInCodeCommentingDevs', 'AffilWithApache',
          'AffilWithSpark','Cherry', 'Git', 'Squash']]
y = data['ReviewDays']



reg = LinearRegression()
reg.fit(x, y)

#print(reg.coef_)
#print(reg.intercept_)
'''
R2 = reg.score(x, y)
print('RSquared is:')
print(R2)

n = x.shape[0]
p = x.shape[1]
AdjustedR2 = 1 - (1-R2)*(n-1)/(n-p-1)
print('Adjusted-R-Squard is:')
print(AdjustedR2)
'''


##Feature Selection

#print('F-statistics and p-values for each independent variable:')
#print(f_regression(x, y))

p_value = f_regression(x, y)[1]
#print(p_value.round(3))


###Creating summary table
regSummary = pd.DataFrame(data=['Size(LOC)', 'NoFiles', 'NoCommits', 'Experience', 'NoComments', 'NoCommentingDevs',
                                'NoAuthorComments', 'NoInCodeCommentingDevs','AffilWithApache','AffilWithSpark',
                                'Cherry', 'Git', 'Squash'], columns=['Features'])
regSummary['Coefficients'] = reg.coef_
regSummary['P-Values'] = p_value.round(3)

print(regSummary)


###Standardization
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)


###Regression with scaled features
scaledReg = LinearRegression()
scaledReg.fit(x_scaled, y)

##Creating Summary table2 (with scaled data)

print('                            ')
print('Intercept/Biasis:                   ' , scaledReg.intercept_)
print('Size(LOC)_Weight:                   ',scaledReg.coef_[0])
print('NoFiles_Weight:                     ',scaledReg.coef_[1])
print('NoCommits_Weight:                   ',scaledReg.coef_[2])
print('Experience_Weight:                  ',scaledReg.coef_[3])
print('NoComments_Weight:                  ',scaledReg.coef_[4])
print('NoCommentingDevs_Weight:            ',scaledReg.coef_[5])
print('NoAuthorComments_Weight:            ',scaledReg.coef_[6])
print('NoInCodeCommentingDevs_Weight:      ',scaledReg.coef_[7])
print('AffilWithApache_Weight:             ',scaledReg.coef_[8])
print('AffilWithSpark_Weight:              ',scaledReg.coef_[9])
print('Cherry_Weight:                      ',scaledReg.coef_[10])
print('Git_Weight:                         ',scaledReg.coef_[11])
print('Squash_Weight:                      ',scaledReg.coef_[12])

'''
SReg_summary = pd.DataFrame([['Intercept/Bias'],['Size(LOC)'], ['NoFiles'], ['NoCommits'], ['Experience'],
                             ['NoComments'], ['NoCommentingDevs'],
                             ['NoAuthorComments'], ['NoInCodeComments'], ['NoInCodeCommentingDevs'],
                             ['NoInCodeAuthorComments'], ['AffilWithApache'], ['AffilWithSpark'],
                             ['Cherry'], ['Git'], ['Squash']], columns=['Features'])
SReg_summary['Weight'] = scaledReg.intercept_, scaledReg.coef_[0], scaledReg.coef_[1], scaledReg.coef_[2],\
                         scaledReg.coef_[3], scaledReg.coef_[5], scaledReg.coef_[6], scaledReg.coef_[7],\
                         scaledReg.coef_[8], scaledReg.coef_[9], scaledReg.coef_[10], scaledReg.coef_[11],\
                         scaledReg.coef_[12], scaledReg.coef_[13], scaledReg.coef_[14]
print(SReg_summary)
'''

print('              ')

R2 = scaledReg.score(x_scaled, y)
print('RSquared is:')
print(R2)

n = x_scaled.shape[0]
p = x_scaled.shape[1]
AdjustedR2 = 1 - (1-R2)*(n-1)/(n-p-1)
print('Adjusted-R-Squard is:')
print(AdjustedR2)






