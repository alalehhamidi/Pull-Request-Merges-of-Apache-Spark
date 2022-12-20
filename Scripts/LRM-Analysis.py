import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


raw_data = pd.read_csv('D:\Alaleh\FinalSparkData.csv')
q = raw_data['Size(LOC)'].quantile(0.95)
data = raw_data[raw_data['Size(LOC)']<q]
#print(data.head())

x = data[['Size(LOC)', 'NoFiles', 'NoCommits', 'Experience', 'NoComments', 'NoCommentingDevs', 'NoAuthorComments',
          'NoInCodeCommentingDevs', 'AffilWithApache', 'AffilWithSpark']]
y = data['IsMerged']


###Scatter Plot
'''
plt.scatter(x1, y, color='C0')
plt.xlabel('Size', fontsize=20)
plt.ylabel('Is Merged', fontsize=20)
plt.show()
'''


###Plot with a regressin line
'''
xp = sm.add_constant(x1)
reg_lin = sm.OLS(y,xp)
result_lin = reg_lin.fit()

plt.scatter(x1, y, color='C0')
y_hat = x1*result_lin.parmas[1]+result_lin.parmas[0]

plt.plot(x1, y_hat, lw=2.5, color='C8')
plt.xlabel('Size', fontsize=20)
plt.ylabel('Is Merged', fontsize=20)
plt.show()
'''

###Plot with Logistic regression curve
#236



###Regression
xp = sm.add_constant(x)
reg_log = sm.Logit(y,xp)
result_log = reg_log.fit()

print(result_log.summary())



###Accuracy of model
cm_df = pd.DataFrame(result_log.pred_table())
cm = np.array(cm_df)
accuracy_train = (cm[0,0]+ cm[1,1])/cm.sum()
print('accuracy is:')
print(accuracy_train)