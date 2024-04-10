#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_column', None)


# In[37]:


df=pd.read_csv('FastagFraudDetection[1].csv')
df.head()


# ### DATA PREPARATION

# In[38]:


df.drop(['FastagID','Transaction_ID','Vehicle_Plate_Number'], axis=1, inplace=True)
df


# In[39]:


df.head()


# In[40]:


df['Timestamp_date']=pd.to_datetime(df['Timestamp']).dt.day
df['Timestamp_month']=pd.to_datetime(df['Timestamp']).dt.month
df['Timestamp_year']=pd.to_datetime(df['Timestamp']).dt.year


# In[41]:


df


# In[42]:


df[['Latitude', 'Longitude']] = df['Geographical_Location'].str.split(',', expand=True)


# In[158]:


df.drop('Geographical_Location',axis=1, inplace=True)


# In[43]:


df['Latitude']


# In[44]:


df['Longitude']


# In[45]:


df['Longitude']=pd.to_numeric(df['Longitude'])
df['Latitude']=pd.to_numeric(df['Latitude'])


# In[47]:


df.drop('Timestamp',axis=1,inplace=True)


# In[48]:


df.info()


# In[49]:


num=df.select_dtypes(include='number')
cat=df.select_dtypes(include='object')


# In[50]:


cat


# In[51]:


cat_cols=[item for item in df.columns if df[item].dtype=='object']
cat_cols


# In[52]:


for item in cat_cols:
    print(item)
    print(f'{df[item].value_counts()}')
    print('\n')


# In[53]:


df['Fraud_indicator'].value_counts()/len(df)*100


# In[54]:


df


# In[55]:


df['Fraud_indicator']=df['Fraud_indicator'].map({'Fraud':1,'Not Fraud':0})


# ### EDA

# In[64]:


df.describe()


# In[68]:


cat


# In[71]:


plt.figure(figsize=(10,6))
sns.countplot(cat['Vehicle_Type'])


# In[72]:


plt.figure(figsize=(10,6))
sns.countplot(cat['TollBoothID'])


# In[73]:


plt.figure(figsize=(10,6))
sns.countplot(cat['Vehicle_Dimensions'])


# In[76]:


plt.figure(figsize=(10,6))
sns.countplot(cat['Fraud_indicator'])


# ### Insights :
# 1.Most number of ravells have passed through toll booth 101, 102, 103</br>
# 2.Vehicle with high dimensions or large vehicles have used the tolls

# In[77]:


df['Fraud_indicator'].value_counts().plot(kind='pie',autopct='%.2f')


# In[84]:


num


# In[91]:


plt.hist(df['Transaction_Amount'])


# In[95]:


plt.hist(df['Vehicle_Speed'],bins=10)


# In[88]:


plt.hist(df['Amount_paid'])


# In[120]:


def dist_plot(df,col):
    plt.figure(figsize=(10,6))
    sns.distplot(df[col])
    plt.title(f'Dist plot for {col}')


# In[113]:


num_cols=['Transaction_Amount',
 'Amount_paid',
 'Vehicle_Speed']


# In[121]:


for item in num_cols:
    dist_plot(num, item)


# In[118]:


def box_plot(df,col):
    plt.figure(figsize=(8,6))
    sns.boxplot(df[col])
    plt.title(f'Box plot for {col}')


# In[119]:


for item in num_cols:
    box_plot(num, item)


# In[127]:


sns.barplot(df['TollBoothID'],df['Transaction_Amount'])


# In[129]:


sns.barplot(df['Lane_Type'],df['Transaction_Amount'])


# In[130]:


sns.barplot(df['Vehicle_Dimensions'],df['Transaction_Amount'])


# In[132]:


sns.barplot(df['Vehicle_Type'],df['Transaction_Amount'])


# In[133]:


sns.boxplot(df['Vehicle_Dimensions'],df['Transaction_Amount'])


# In[136]:


df


# In[144]:


pd.crosstab(df['Lane_Type'],df['Fraud_indicator'])


# In[147]:


sns.heatmap(pd.crosstab(df['Lane_Type'],df['Fraud_indicator']))


# In[148]:


pd.crosstab(df['Vehicle_Dimensions'],df['Fraud_indicator'])


# In[149]:


sns.heatmap(pd.crosstab(df['Vehicle_Dimensions'],df['Fraud_indicator']))


# In[150]:


corr=df.corr()


# In[153]:


plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True)


# In[159]:


df


# In[160]:


df=pd.get_dummies(df,columns=['Vehicle_Type','TollBoothID','Lane_Type','Vehicle_Dimensions'],drop_first=True)


# In[161]:


df


# In[164]:


from sklearn.feature_selection import VarianceThreshold
var_thres=VarianceThreshold(threshold=0)
var_thres.fit(df)


# In[165]:


var_thres.get_support()


# In[168]:


df.drop('Timestamp_year',axis=1, inplace=True)


# In[169]:


corr=df.corr()
plt.figure(figsize=(20,12))
sns.heatmap(corr, annot=True)


# In[170]:


def correlation(df, threshold):
    col_corr=set()
    corr_matrix=df.corr()
    for i in range (len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
                


# In[172]:


X=df.drop('Fraud_indicator',axis=1)
y=df['Fraud_indicator']


# In[173]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)


# In[176]:


corr_features=correlation(X_train, 0.8) 


# In[177]:


corr_features


# In[178]:


from sklearn.preprocessing import StandardScaler


# In[181]:


df['Amount_difference']=df['Transaction_Amount']-df['Amount_paid']


# In[184]:


df


# In[193]:


df1


# In[194]:


df1=df1.drop('Amount_difference',axis=1)


# In[196]:


from sklearn.model_selection import train_test_split
X=df1.drop('Fraud_indicator',axis=1)
y=df1['Fraud_indicator']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)


# In[198]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[199]:


X_train = scaler.fit_transform(X_train)

# Transform the testing data using the scaler fitted on the training data
X_test = scaler.transform(X_test)


# In[201]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[202]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[203]:


y_pred = model.predict(X_test)


# In[204]:


y_pred


# In[205]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[206]:


y_pred_train = model.predict(X_train)

y_pred_train


# In[207]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[220]:


df2.drop(['Amount_paid','Amount_difference'],axis=1,inplace=True)


# In[221]:


from sklearn.model_selection import train_test_split
X=df2.drop('Fraud_indicator',axis=1)
y=df2['Fraud_indicator']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)


# In[222]:


X_train = scaler.fit_transform(X_train)

# Transform the testing data using the scaler fitted on the training data
X_test = scaler.transform(X_test)


# In[223]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[224]:


y_pred = model.predict(X_test)


# In[225]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[226]:


y_pred_train = model.predict(X_train)

y_pred_train


# In[227]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[ ]:




