import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

raw_data = pd.read_csv('D:\Alaleh\FinalSparkData.csv')
q = raw_data['Size(LOC)'].quantile(0.95)
data = raw_data[raw_data['Size(LOC)']<q]

###Checking OLS assumptions

#plt.scatter(data['Size(LOC)'], data['ReviewDays'])
#plt.scatter(data['NoFiles'], data['ReviewDays'])
#plt.scatter(data['NoCommits'], data['ReviewDays'])
#plt.scatter(data['Experience'], data['ReviewDays'])
#plt.scatter(data['NoComments'], data['ReviewDays'])
#plt.scatter(data['NoCommentingDevs'], data['ReviewDays'])
#plt.scatter(data['NoAuthorComments'], data['ReviewDays'])
#plt.scatter(data['NoInCodeComments'], data['ReviewDays'])
#plt.scatter(data['NoInCodeCommentingDevs'], data['ReviewDays'])
#plt.scatter(data['NoInCodeAuthorComments'], data['ReviewDays'])
#plt.scatter(data['AffilWithApache'], data['ReviewDays'])

#plt.show()

log_reviewDays = np.log(data['ReviewDays'])
data['logtime'] = log_reviewDays

#plt.scatter(data['Size(LOC)'], data['logtime'])
#plt.scatter(data['NoFiles'], data['logtime'])
#plt.scatter(data['NoCommits'], data['logtime'])
#plt.scatter(data['Experience'], data['logtime'])
#plt.scatter(data['NoComments'], data['logtime'])
#plt.scatter(data['NoCommentingDevs'], data['logtime'])
#plt.scatter(data['NoAuthorComments'], data['logtime'])
#plt.scatter(data['NoInCodeComments'], data['logtime'])
#plt.scatter(data['NoInCodeCommentingDevs'], data['logtime'])
#plt.scatter(data['NoInCodeAuthorComments'], data['logtime'])
#plt.scatter(data['AffilWithApache'], data['logtime'])

#plt.show()

#By comparing plots before and after LOG-Transformatin, we realize that the
#relation betwean each independent variable and dependent variable is more linear befor LOG transformation




### Checking Multicollinirity

variables = data[['Size(LOC)', 'NoFiles', 'NoCommits']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
print(vif)

variables = data[['NoCommentingDevs', 'NoInCodeCommentingDevs']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
print(vif)


variables = data[['NoAuthorComments', 'NoInCodeAuthorComments']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
print(vif)


variables = data[['NoComments', 'NoInCodeComments']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
print(vif)

# So we should eliminate columns 'NoInCodeAuthorComments' and 'NoInCodeComments' from our data.