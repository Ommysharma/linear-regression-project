#!/usr/bin/env python
# coding: utf-8

# In[1953]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score



# In[1954]:


boom_data =pd.read_csv('day.csv')


# In[1955]:


boom_data


# ###  Step 1: Reading and Understanding of data 

# In[1956]:


# check shape of data so Here 730 row and 16 columns

boom_data.shape


# In[1957]:


#data type information ,here Nan value is not present

boom_data.info()


# In[1958]:


#describe the data

boom_data.describe()


# In[1959]:


# check the columns

boom_data.columns


# In[1960]:


boom_data.head()


# In[1961]:


# We don't need the Feature 'instant',it is just serial number of the dataset.
# We don't need the Feature 'dtday' because it is already expalained by the other features like: yr, mnth, weekday, workingday and holiday.

boom_data.drop(['instant','dteday'],axis=1,inplace=True)


# In[1962]:


#making a heatmap to showcase correlation between the variables and decide whether we can perform linear regression on the dataset
plt.figure(figsize=(20, 12))
sns.heatmap(boom_data.corr(), cmap='OrRd', annot=True)
plt.title('Correlation between variables in the dataset')
plt.show()


# ### As we can see, there are several variables correlated to count variable (cnt) such as registered, casual, temp, atemp, yr, workingday, we can thereby conduct a linear regresssion model

# In[1963]:


boom_data[boom_data['season']==4]


# In[ ]:





# In[1964]:


# converting yr boolean column to year data

def yr_assing(years):
    if years==0:
        return '2018'
    else:
        return '2019'
    
boom_data['year']=boom_data['yr'].apply(yr_assing)


# In[1965]:


boom_data


# In[1966]:


# converting month number column to month name

def month_name(monthNo):
 
   if monthNo == 1:
       return 'Jan'
   elif monthNo == 2:
       return 'Feb' 
   elif monthNo == 3:
       return 'Mar'
   elif monthNo == 4:
       return 'Apr'
   elif monthNo== 5:
       return 'May' 
   elif monthNo ==6:
       return 'Jun'
   elif monthNo == 7:
       return 'Jul' 
   elif monthNo == 8:
       return 'Aug' 
   elif monthNo == 9:
       return 'Sep'
   elif monthNo == 10:
       return 'Oct'
   elif monthNo == 11:
       return 'Nov' 
   elif monthNo == 12:
       return 'Dec'
   
   else:
       return None
boom_data['mnth']=boom_data['mnth'].apply(month_name)


# In[1967]:


boom_data


# In[1968]:


# converting weakday number into weakdayname

def weakday_name(weakdayNo):
  
    if weakdayNo == 0:
        return 'Sun'
    elif weakdayNo == 1:
        return 'Mon' 
    elif weakdayNo == 2:
        return 'Tue'
    elif weakdayNo == 3:
        return 'Wed'
    elif weakdayNo== 4:
        return 'Thurs' 
    elif weakdayNo ==5:
        return 'Fri'
    elif weakdayNo == 6:
        return 'Sat' 
    
    else:
        return None
boom_data['weekday']=boom_data['weekday'].apply(weakday_name)


# In[1969]:


boom_data


# In[1970]:


#mapping categorical variables with their subcategories to help with visualization analysis 

boom_data['season']=boom_data['season'].map({1: 'spring', 2: 'summer',3:'fall', 4:'winter' })


# In[1971]:


boom_data['weathersit']=boom_data.weathersit.map({1: 'Clear',2:'Mist + Cloudy',3:'Light Snow',4:'Snow + Fog'})


# In[1972]:


boom_data['holiday']=boom_data.holiday.map({0: 'Holiday',1:'Not Holiday'})


# In[1973]:


boom_data['workingday']=boom_data.workingday.map({0: 'workingday',1:'Not workingday'})


# In[1974]:


boom_data


# In[1975]:


boom_data.info()


# In[1976]:


#visualizing the categorical variables of the dataset using boxplot 
data =boom_data.select_dtypes(include = "object").columns.to_list()
catlist=[]
for k in data:
    catlist.append(k)
print(catlist)

def boxplotdata(catcol,count):
    print(count)
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 4, count)
    sns.boxplot(x=catcol, y='cnt', data=boom_data)
    plt.show()
        
        


# In[1977]:


for index, value in enumerate(catlist, start=1):
    boxplotdata(value,index)


# ### These are some observation over box plot
# ### 1. Bike rentals peak during the summer and fall months.
# ### 2 .September and October see the highest bike rental rates.
# ### 3 .Bike rentals are most popular during clear weather conditions.
# ### 4 .Saturdays, Wednesdays, and Thursdays are the busiest days for bike rentals.
# ### 5 .Bike rental numbers were higher in 2019.
# ### 6 .Bike rental rates do not significantly vary between weekdays and weekends.
# ### 7 .Bike rental rates increase on holidays.
# 

# In[1978]:


#dropping the un-required variables from the dataset 
#dropping the variables atemp,year casual, registered as they are not needed for the analysis 
#atemp is not needed as temp is already being used, dteday and casual are also not required for regression analysis 

df = boom_data.drop(['atemp','year','casual', 'registered'], axis=1)
df.head(5)


# In[1979]:


#checking the type, info, description, missing values, shape of the dataset
df.shape


# In[1980]:


df.describe()


# In[1981]:


df.info()


# In[1982]:


#creating dummy variables 
#creating dummy variables for the variables of month, weekday, weathersit, seasons
month = pd.get_dummies(df.mnth, drop_first=True)
weekday = pd.get_dummies(df.weekday, drop_first=True)
weathersit = pd.get_dummies(df.weathersit, drop_first=True)
season = pd.get_dummies(df.season, drop_first=True)


# In[1983]:


#adding the dummy variables to the original dataframe
df = pd.concat([df,month, weekday, weathersit, season], axis=1)
df.head(5)


# In[1984]:


# map bool into integer value.
bool_cols =df.select_dtypes(include = "bool").columns.to_list()
print(data)
for col in bool_cols:
    df[col] = df[col].map({True: 1,False :0 })


# In[1985]:


df


# In[1986]:


# dropping the variables season,mnth,weekday,weathersit as we have created the dummies for it
df.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True)
df.head(100)


# In[1987]:


df.head(100)


# In[1988]:


df.info()


# In[1989]:


# map object into integer value.
bool_cols =df.select_dtypes(include = "object").columns.to_list()
print(data)
for col in bool_cols:
    if col == 'holiday':
        
        df[col] = df[col].map({'Holiday': 0,'Not Holiday':1})
    elif col == 'workingday':
        df[col] = df[col].map({'workingday': 0,'Not workingday':1})


# In[1990]:


df


# In[1991]:


#making a heatmap to showcase correlation between the new variables 
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), cmap='YlOrBr', annot=True)
plt.title('Correlation between variables in the dataset after data preparation is done')
plt.show()


# ### Step 2: Preparing the data for model training (train-test split, rescaling)

# ### Feature Scaling continuous variables
# ### To make all features in same scale to interpret easily
# 
# ### Following columns are continous to be scaled
# ### temp,hum,windspeed

# In[1992]:


# Splitting the Data into Training and Testing Sets
df_train,df_test = train_test_split(df,train_size=0.7,random_state=100)
print(df_train.shape)
print(df_test.shape)


# In[1993]:


#we have to rescale the variables like hum, temp, windspeed, cnt as they have large values as compared to the other variables of the dataset
#we have to normalize these values using the scaler.fit_transform() 
scaler = MinMaxScaler()
scaler_var = ['hum', 'windspeed', 'temp', 'cnt']
df_train[scaler_var] = scaler.fit_transform(df_train[scaler_var])


# In[1994]:


#checking the normalized values of the train set after performing scaling 
df_train.describe()


# In[1995]:


#checking the correlation coefficients to see which variables are highly correlated post data preparation and rescaling

plt.figure(figsize = (20, 10))
sns.heatmap(df_train.corr(), cmap="YlGnBu", annot=True)
plt.title('Heatmap to check correlation after data preparation and rescaling')
plt.show()


# ### here we can see over heat map the correlation between cnt and temp is high.

# In[1996]:


# here using pairplot we can check the varience between the temp and cnt.

plt.figure(figsize=[5,5])
plt.scatter(df_train.temp, df_train.cnt)
plt.title('Correlation between count vs temp')
plt.show()


# In[1997]:


#building the first model using the variable temp
#preparing the variables for model building 

y_train_set = df_train.pop('cnt')
X_train_set = df_train


# In[1998]:


# check the varibale of y_train_set
y_train_set


# In[1999]:


# check the varibale of x_train_set
X_train_set


# In[2000]:


#add a constant (intercept)
X_train_sm = sm.add_constant(X_train_set['temp'])
print(X_train_sm)

#create first model
lrm = sm.OLS(y_train_set, X_train_sm)

#fit
lrm_model = lrm.fit()

#params
lrm_model.params


# In[2001]:


#checking model summary 
lrm_model.summary()


# In[2002]:


#add one more independent variable
X_train_sm = sm.add_constant(X_train_set[['temp','yr']])
print(X_train_sm)

#create model
lr = sm.OLS(y_train_set, X_train_sm)

#fit
lr_model = lrm.fit()

#params
lr_model.params


# In[2003]:


lr_model.summary()


# ### we can see that the r2 value is 41% and p value for tis model is 000.

# ### Model check for all independent variable.

# In[2004]:


X_train_set.columns.to_list()


# In[2005]:


#add one more independent variable
X_train_sm = sm.add_constant(X_train_set[X_train_set.columns.to_list()])


#create model
lr = sm.OLS(y_train_set, X_train_sm)

#fit
lr_model = lrm.fit()

#params
lr_model.params


# In[2006]:


#check summary

lr_model.summary()


# 
# ### strong multicollinearity problems or that the design matrix is singular.
# ### R-squared is now .850 which means 85% of the variance in the count quantity is explained with all the variables
# 
# ### Using RFE:
# ### We have found out the R-squared values and the best fit lines using the manual approach for feature selection. We will now using the automated approach for selecting the values required for building the most optimized models and dropping the values which are redundant for our model building approach. We will now use RFE for automated approach, along with VIF to determine the variables to drop.

# In[ ]:





# In[2007]:


#creating the RFE object
lm = LinearRegression()
lm.fit(X_train_set, y_train_set)

#setting feature selection variables to 15
rfe = RFE(lm, n_features_to_select = 15) 

#fitting rfe ofject on our training dataset
rfe = rfe.fit(X_train_set, y_train_set)


# In[2008]:


rfe


# In[2009]:


#Columns selected by RFE and their weights
list(zip(X_train_set.columns,rfe.support_,rfe.ranking_))


# In[2010]:


#getting the selected feature variables in one one variable
actual_rfe = X_train_set.columns[rfe.support_]


# In[2011]:


actual_rfe


# In[2012]:


# Features Rejected by the RFE
rejected_rfe = X_train_set.columns[~(rfe.support_)]
rejected_rfe


# In[2013]:


#adding constant to training variable
X_train_rfe = sm.add_constant(X_train_set[actual_rfe])

#creating first training model with rfe selected variables
lr = sm.OLS(y_train_set, X_train_rfe)

#fit
lr_model = lr.fit()

#params
lr_model.params


# In[2014]:


lr_model.summary()


# In[2015]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
# X_train_ref = X_train_rfe.drop(['const'], axis = 1)


# In[2016]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[2017]:


#checking the VIF of the model 

#dropping the spring variables from the dataset
X_train_ref_sun = X_train_rfe.drop(['spring'], axis = 1)


# In[2018]:


#adding constant to training variable
X_train_rfe_sun = sm.add_constant(X_train_ref_sun)

#creating first training model with rfe selected variables
lr = sm.OLS(y_train_set, X_train_rfe_sun)

#fit
lr_model = lr.fit()

#params
lr_model.params


# In[2019]:


lr_model.summary()


# In[2020]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_rfe_sun
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[2021]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
X_train_final = X_train_rfe_sun.drop(['const'], axis = 1)


# ### Step 3: Residual analysis
# 

# In[2022]:


X_train_final = sm.add_constant(X_train_final)

X_train_final


# In[2023]:


y_train_pred


# In[2024]:


res = y_train_set - y_train_pred
#distribution of the error terms shown here; distribution should be centered around 0 and should be a normal distribution
sns.distplot(res)
plt.title('Error distribution')
plt.show()


#  ###  Step 4 Prediction and Evaluation on the test set.

# In[2025]:


df_test


# In[2026]:


#we have to rescale the variables like hum, temp, windspeed, cnt as they have large values as compared to the other variables of the dataset
#we have to normalize these values using the scaler.fit_transform() 

num_vars = ['hum', 'windspeed', 'temp', 'cnt']

df_test[num_vars]=scaler.transform(df_test[num_vars])

y_test = df_test.pop('cnt')
X_test = df_test
X_test_sm = sm.add_constant(X_test)
X_test_sm.head()


# In[2027]:


X_test_sm.describe()


# In[ ]:





# In[2028]:


rejected_rfe.to_list()


# In[2029]:


X_test_sm = X_test_sm.drop(['workingday',
 'Aug',
 'Feb',
 'Jun',
 'Mar',
 'May',
 'Oct',
 'Mon',
 'Sat',
 'Sun',
 'Thurs',
 'Tue',
 'Wed','spring'],axis=1)

# Predict the Price for Test Data using the Trainned Model
y_test_pred = lr_model.predict(X_test_sm)

y_test_pred.sample(5)


# In[2030]:


#r2 score of the test set
r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
print('r2 score on the test set is', r2_test)


# In[2031]:


#r2 score of the training set
r2_train = r2_score(y_true=y_train_set, y_pred= y_train_pred)
print('r2 score on the train set is', r2_train)


# In[2036]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
sns.regplot(x=y_test, y=y_test_pred, ci=52, fit_reg=True, line_kws={"color": "black"})
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_pred', fontsize = 16)               
plt.xlabel('y_test', fontsize = 14)                          
plt.ylabel('y_pred', fontsize = 14) 
plt.show()


# ### The Final Model accuray is around 80.0%, Which is a Good Score.
# ### The Model accuracy difference between the Train and Test Data is less than 3% which is acceptable.

# In[2038]:


Cofficients_test = round(lr_model.params,2)
data_test = Cofficients.sort_values(ascending = False)
data_test


# In[2039]:


for i in range(0, len(data_test)):
    print(data_test[i],'*',data_test.index[i],end ='')
    print(' + ',end ='')


# #### Temperature is a key factor positively impacting business, while other environmental conditions like rain, humidity, wind speed, and cloudiness negatively affect it. The company must develop strategies to mitigate the effects of adverse weather, particularly rain, on users. Bike demand peaks in winter and summer, with a noticeable decline during rainy days and seasons.
# 
# ####  Interestingly, bike rentals see a significant surge on Saturdays compared to other weekdays. Demand for bike rentals grew substantially from 2018 to 2019, indicating Boom Bikes' successful market penetration. The company can anticipate a 19% business growth without additional investment.

# In[ ]:




