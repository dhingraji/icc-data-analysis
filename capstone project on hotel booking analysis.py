# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:21:16 2022

@author: pc1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick


pd.options.display.max_columns=None
data=pd.read_csv('hotel_bookings.csv')
data.head()
df=data.copy()
df.isnull().sum().sort_values(ascending=False)[:10]
df[['agent','company']]=df[['agent','company']].fillna(0.0)
df['country'].fillna(data.country.mode().to_string(),inplace=True)
df['children'].fillna(round(data.children.mean()),inplace=True)
df[(df.adults+df.babies+df.children)==0].shape
df=df.drop([(df.adults+df.babies+df.children)==0].index)
df.dtypes
df[['children','agent','company']]=df[['children','agent','company']].astype('int64')
# bookings were cancelled
def get_count(series,limit=None):
    
    '''
    INPUT:
       series: Pandas Series(Single Column for DataFrame)
       limit:If value given,limit the output value to first limit samples.
   OUTPUT:
       x=Unique values
       y= Count of unique values
    '''
     if limit != None:
        series = series.value_counts()[:limit]
    else:
        series = series.value_counts()
         
    x=series.index()
    y=series/series.sum()*100
    
    return x.values,y.values
x,y=get_count(df['is_canceled'])
def plot(x,y,x_label=None,y_label=None,title=None,figsize=(7,5),type='bar'):
    
    '''
    INPUT:
       x:       Array containing values for x_axis
       y:       Array containing values for y_axis
       x_label: String value for x_axis label
       y_label: String value for y_axis label
       title:   String value for plot title
       figsize: tuple value , for figure size
       type:    type of plot (default is bar plot)
   OUTPUT:
       Display the plot
      '''
      sns.set_style('darkgrid')
      fig,ax=plt.subplots(figsize=figsize)
      ax.yaxis.set_major_formatter(mtick.PercentFormatter())
      
      if x_label!=None:
          ax.set_xlabel(x_label)
          
      if y_label!=None:
          ax.set_ylabel(y_label)
    
      if title!=None:
          ax.set_title(title)
      
      if type=='bar':
          sns.barplot(x,y,ax=ax)
     elif type=='line':
         sns.lineplot(x,y,ax=ax,sort=False)
     
        
    plt.show()
    
plot(x,y,x_label='Booking Cancelled (No=0,Yes=1)',y_label='Booking (%)')
df_not_canceled=df[df['is_canceled']==0]
# Booking ratio
x,y=get_count(df_not_canceled['hotel'])
plot(x,y,x_label='Hotels',y_label='Total Bookings (%)',title='Hotel Comparison')
# Percentage of booking each year
x,y=get_count(df_not_canceled['arrival_date_year'])
plot(x,y,x_label='Year',y_label='Total Booking (%)',title='Year Comparison')
plt.subplot(figsize=(7,5))
sns.countplot(x='arrival_date_year',hue='hotel',data=df_not_canceled);
# Busiest month for hotels
new_order=['January','Feburary','March','April','May','June','July','August','September',
           'October','November','December']
sorted_months = df_not_canceled['arrival_date_month'].value_counts().reindex(new_order)

x = sorted_months.index
y = sorted_months/sorted_months.sum()*100


#sns.lineplot(x, y.values)
plot(x, y.values, x_label='Months', y_label='Booking (%)', title='Booking Trend (Monthly)', type='line', figsize=(18,6))
## Order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 
             'November', 'December']


## Select only City Hotel
sorted_months = df_not_canceled.loc[df.hotel=='City Hotel' ,'arrival_date_month'].value_counts().reindex(new_order)

x1 = sorted_months.index
y1 = sorted_months/sorted_months.sum()*100



## Select only Resort Hotel
sorted_months = df_not_canceled.loc[df.hotel=='Resort Hotel' ,'arrival_date_month'].value_counts().reindex(new_order)

x2 = sorted_months.index
y2 = sorted_months/sorted_months.sum()*100






## Draw the line plot

fig, ax = plt.subplots(figsize=(18,6))

ax.set_xlabel('Months')
ax.set_ylabel('Booking (%)')
ax.set_title('Booking Trend (Monthly)')


sns.lineplot(x1, y1.values, label='City Hotel', sort=False)
sns.lineplot(x1, y2.values, label='Resort Hotel', sort=False)

plt.show()
# from which country most guests come
import pycountry as pc

x,y = get_count(df_not_canceled['country'], limit=10)

# For each country code select the country name 
country_name = [pc.countries.get(alpha_3=name).name for name in x]

plot(country_name,y, x_label='Countries', y_label='Total Booking (%)', title='country-wise comparison', figsize=(15,7))
# how long the people stay in the hotel 
total_nights = df_not_canceled['stays_in_weekend_nights']+ df_not_canceled['stays_in_week_nights']
x,y = get_count(total_nights, limit=10)

plot(x,y, x_label='Number of Nights', y_label='Booking Percentage (%)', title='Night Stay Duration (Top 10)', figsize=(10,5))              
df_not_canceled.loc[:,'total_nights'] = df_not_canceled['stays_in_weekend_nights']+ df_not_canceled['stays_in_week_nights']

fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlabel('No of Nights')
ax.set_ylabel('No of Nights')
ax.set_title('Hotel wise night stay duration (Top 10)')
sns.countplot(x='total_nights', hue='hotel', data=df_not_canceled,
              order = df_not_canceled.total_nights.value_counts().iloc[:10].index, ax=ax);
# Select single, couple, multiple adults and family
single   = df_not_canceled[(df_not_canceled.adults==1) & (df_not_canceled.children==0) & (df_not_canceled.babies==0)]
couple   = df_not_canceled[(df_not_canceled.adults==2) & (df_not_canceled.children==0) & (df_not_canceled.babies==0)]
family   = df_not_canceled[df_not_canceled.adults + df_not_canceled.children + df_not_canceled.babies > 2]


# Make the list of Category names, and their total percentage
names = ['Single', 'Couple (No Children)', 'Family / Friends']
count = [single.shape[0],couple.shape[0], family.shape[0]]
count_percent = [x/df_not_canceled.shape[0]*100 for x in count]


#Draw the curve
plot(names,count_percent,  y_label='Booking (%)', title='Accommodation Type', figsize=(10,7))
df_subset = df.copy()
# Make the new column which contain 1 if guest received the same room which was reserved otherwise 0
df_subset['Room'] = 0
df_subset.loc[ df_subset['reserved_room_type'] == df_subset['assigned_room_type'] , 'Room'] = 1


# Make the new column which contain 1 if the guest has cancelled more booking in the past
# than the number of booking he did not cancel, otherwise 0

df_subset['net_cancelled'] = 0
df_subset.loc[ df_subset['previous_cancellations'] > df_subset['previous_bookings_not_canceled'] , 'net_cancelled'] = 1  
# Remove the less important features
df_subset = df_subset.drop(['arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
                            'arrival_date_month','assigned_room_type','reserved_room_type','reservation_status_date',
                            'previous_cancellations','previous_bookings_not_canceled'],axis=1)     
# Remove reservation_status column
# because it tells us if booking was cancelled 
df_subset = df_subset.drop(['reservation_status'], axis=1)  
# Plot the heatmap to see correlation with columns
fig, ax = plt.subplots(figsize=(22,15))
sns.heatmap(df_subset.corr(), annot=True, ax=ax);
# Convert  categorical variable into numerical variable
def transform(dataframe):
    
    
    ## Import LabelEncoder from sklearn
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    
    
    ## Select all categorcial features
    categorical_features = list(dataframe.columns[dataframe.dtypes == object])
    
    
    ## Apply Label Encoding on all categorical features
    return dataframe[categorical_features].apply(lambda x: le.fit_transform(x))

df = transform(df)   
# Train Test spilit
def data_split(df, label):
    
    from sklearn.model_selection import train_test_split

    X = df.drop(label, axis=1)
    Y = df[label]

    x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=0)
    
    return x_train, x_test, y_train, y_test



x_train, x_test, y_train, y_test = data_split(df_subset, 'is_canceled')     
# Decision Tree
def train(x_train, y_train):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train,y_train)
    
    return clf


clf = train(x_train, y_train)
# Evaluation of model
def Score(clf,x_train,y_train,x_test,y_test):
    train_score = clf.score(x_train,y_train)
    test_score = clf.score(x_test,y_test)

    print("========================================")
    print(f'Training Accuracy of our model is: {train_score}')
    print(f'Test Accuracy of our model is: {test_score}')
    print("========================================")
    
    
Score(clf,x_train,y_train,x_train,y_train)
# Getting Prediciton of 10th record of x_train
prediction = clf.predict(x_train.iloc[10].values.reshape(1,-1))

## Actual Value of 10th record of x_train from y_train
actual_value = y_train.iloc[10]

print(f'Predicted Value \t: {prediction[0]}')
print(f'Actual Value\t\t: {actual_value}')