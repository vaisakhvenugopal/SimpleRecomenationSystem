#!/usr/bin/env python
# coding: utf-8

# <h1><center>BeerMart : RECOMMENDATION SYSTEM</center></h1>

# ## Problem Statment
# 
# - Crate a collaborative filtering recommentation system
#     - Data preparation (tranformation, selction & EDA)
#     - Model building (user based & item based)
#     - Model compariasion

# In[1]:


#Loading required libraries
get_ipython().run_line_magic('matplotlib', 'inline')

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
import seaborn as sns


# In[2]:


#Geting the data loaded to pandas dataframe
file_path = './beer_data.csv'
data = pd.read_csv(file_path)
data.shape


# In[3]:


#Checking the data using head() method
data.head()


# In[4]:


#Removing the duplicates using drop_duplicates() method
data_1 = data.drop_duplicates(subset= ['beer_beerid' , 'review_profilename'], keep="last")
data_1.shape


# ### Q: Figure out an appropriate value of N using EDA; this may not have one correct answer, but you should not choose beers that have an extremely low number of ratings.

# In[5]:


#Creating a dataframe with beer review counts.
beer_review_counts = pd.DataFrame(data_1.pivot_table(index=['beer_beerid'], values = ['review_overall'],
                                                     aggfunc = {'review_overall' : len}))


# In[6]:


#Sorting the counts asendig order.
beer_review_counts.sort_values("review_overall", axis = 0, ascending = True,inplace = True, na_position ='last')
beer_review_counts.reset_index(inplace= True)


# In[7]:


#Ploting the overall spread of the beer review counts
plt.figure(figsize=(18,6)) # 10 is width, 7 is height
plt.plot(beer_review_counts['review_overall'])  # green dots
plt.legend(loc='best')
plt.show()


# In[8]:


#Ploting the lower range of reviews to better decide the cut-off 
plt.figure(figsize=(18,6)) # 10 is width, 7 is height
plt.plot(beer_review_counts['review_overall'][:35000])  # green dots
plt.legend(loc='best')
plt.show()


# In[18]:


#Ploting histogram for better visibility of the lower range of beer review count upto 300
plt.figure(figsize=(18,6))
bins = [ 10, 20, 30, 50, 100, 150, 180, 195, 205, 220, 250, 300]
plt.hist(beer_review_counts['review_overall'], bins, rwidth=0.8)
plt.show()


# In[12]:


#Ploting histogram for better visibility of the lower range of beer review count upto 10
plt.figure(figsize=(18,6)) 
bins = [1, 2, 3, 4, 5, 10]
plt.hist(beer_review_counts['review_overall'], bins, rwidth=0.8)
plt.show()


# ### Based on the able plots, it cut of take an minimum 2 reviews. A separate dataset will be created witht he cut-off of 2 ratings minimum. In adition reffer the below rating counts also to back this cut-off.

# ### Q What are the unique values of ratings?

# In[13]:


#Summrising the unique rating values & counts of the same
data_1.review_overall.value_counts()


# ### Rating range is from 0 to 5, with 0.5 intervel. Most number of ratings is 4 and the leaset are 0 to 1.5.

# ## Q: Visualise - The average beer ratings

# In[14]:


#Getting the count of review with respect to the beer
beer_review_avg = pd.DataFrame(data_1.pivot_table(index=['beer_beerid'], values = ['review_overall'],
                                                     aggfunc = {'review_overall' : np.mean}))


# In[15]:


#Sort review with respect to the beer rating counts
beer_review_avg.sort_values("review_overall", axis = 0, ascending = True,inplace = True, na_position ='last')
beer_review_avg.reset_index(inplace= True)


# In[19]:


#Ploting the overall numbers of rating in asenting order
plt.figure(figsize=(18,6)) 
plt.plot(beer_review_avg['review_overall'],'o', linewidth=2)
plt.legend(loc='best')
plt.show()


# In[20]:


#Creating a histogram of review counts w.r.t beer
plt.figure(figsize=(18,6))
plt.hist(beer_review_avg['review_overall'], histtype='bar', rwidth=0.8)
plt.show()


# ## Q: Visualise - The average user ratings

# In[21]:


#Create the rating counts w.r.t user.
user_review_avg = pd.DataFrame(data_1.pivot_table(index=['review_profilename'], values = ['review_overall'],
                                                     aggfunc = {'review_overall' : np.mean}))


# In[22]:


#Sorting the user based rating count
user_review_avg.sort_values("review_overall", axis = 0, ascending = True,inplace = True, na_position ='last')
user_review_avg.reset_index(inplace= True)


# In[23]:


#Ploting the overall rating w.r.t user
plt.figure(figsize=(18,6)) 
plt.plot(user_review_avg['review_overall'],'o', linewidth=2)
plt.legend(loc='best')
plt.show()


# In[25]:


#Ploting histogram for the rating counts w.r.t user
plt.figure(figsize=(18,6)) 
plt.hist(user_review_avg['review_overall'],histtype='bar', rwidth=0.8)
plt.show()


# ### Observations
# - Overall rating counts are centered around 4. Both in the case of user based & beer baseed
# - There is no abnormal terend visible in the rating count
# - 0-2 rating counts are less. For modeling greated than 2 reviews is considered

# # Q: Recommendation Models

# In[54]:


#Creating a list of beers to be removed from the data set, which have below 2 rating
beer_review_counts = pd.DataFrame(data_1.pivot_table(index=['beer_beerid'], values = ['review_overall'],
                                                     aggfunc = {'review_overall' : len}).reset_index())
beer_exclusion_list = list(beer_review_counts.loc[beer_review_counts['review_overall'] <= 2].beer_beerid)
len(beer_exclusion_list)


# In[95]:


#Creating a list of users to be removed from the data set, which have below 2 rating
user_review_counts = pd.DataFrame(data_1.pivot_table(index=['review_profilename'], values = ['review_overall'],
                                                     aggfunc = {'review_overall' : len}).reset_index())
user_exclusion_list = list(user_review_counts.loc[user_review_counts['review_overall'] <= 2].review_profilename)
len(user_exclusion_list)


# In[96]:


#Removing the beer ids not required for the evaluations
data_2 = data_1[~data_1.beer_beerid.isin(beer_exclusion_list)]
data_2.shape


# In[97]:


#Removing the user ids not required for the evaluations
data_3 = data_2[~data_2.review_profilename.isin(user_exclusion_list)]
data_3.shape


# In[98]:


print('% data remaining = ' + str((data_3.shape[0]/data_1.shape[0])*100))


# In[99]:


#Test- Train Split


# In[100]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data_3, test_size=0.30, random_state=31)


# In[101]:


print(train.shape)
print(test.shape)


# In[102]:


# pivot ratings into beer features
df_beer_features = train.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
).fillna(0)


# In[103]:


df_beer_features.head()


# In[104]:


#dummy Data set creation for future use
dummy_train = train.copy()
dummy_test = test.copy()

dummy_train['review_overall'] = dummy_train['review_overall'].apply(lambda x: 0 if x>=1 else 1)
dummy_test['review_overall'] = dummy_test['review_overall'].apply(lambda x: 1 if x>=1 else 0)


# In[105]:


# The beers not rated by user is marked as 1 for prediction. 
dummy_train = dummy_train.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
).fillna(1)

# The beers not rated by user is marked as 0 for evaluation. 
dummy_test = dummy_test.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
).fillna(0)


# In[106]:


dummy_train.head()


# In[107]:


dummy_test.head()


# In[108]:


#Transformtation


# In[109]:


#Using Consine Similarity
from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(df_beer_features, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[110]:


user_correlation.shape


# In[111]:


#Using Adjusted Consine Similarity
beer_features = train.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
)


# In[112]:


beer_features.head()


# In[113]:


#Normalizing the rating of the beer for each user aroung 0 mean
mean = np.nanmean(beer_features, axis=1)
df_subtracted = (beer_features.T-mean).T


# In[114]:


df_subtracted.head()


# In[115]:


#Finding the cosine similarity
from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[116]:


#Predictions
user_correlation[user_correlation<0]=0
user_correlation


# In[117]:


user_predicted_ratings = np.dot(user_correlation, beer_features.fillna(0))
user_predicted_ratings


# In[118]:


user_predicted_ratings.shape


# In[119]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# In[120]:


#Finding to 10 recommentatin for user 1
user_final_rating.iloc[1].sort_values(ascending=False)[0:10]


# ### Item Based Similarity

# In[121]:


beer_features = train.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
).T

beer_features.head()


# In[122]:


# Normalising the beer rating for each beer
mean = np.nanmean(beer_features, axis=1)
df_subtracted = (beer_features.T-mean).T


# In[123]:


df_subtracted.head()


# In[124]:


from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[125]:


item_correlation[item_correlation<0]=0
item_correlation


# In[126]:


#Prediction
item_predicted_ratings = np.dot((beer_features.fillna(0).T),item_correlation)
item_predicted_ratings


# In[127]:


item_predicted_ratings.shape


# In[128]:


dummy_train.shape


# In[129]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()


# In[130]:


#Top 10 prediciton for the user1
item_final_rating.iloc[1].sort_values(ascending=False)[0:10]


# ### Evaluation

# In[132]:


#using User Similarity
test_beer_features = test.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
)
mean = np.nanmean(test_beer_features, axis=1)
test_df_subtracted = (test_beer_features.T-mean).T

# User Similarity Matrix
test_user_correlation = 1 - pairwise_distances(test_df_subtracted.fillna(0), metric='cosine')
test_user_correlation[np.isnan(test_user_correlation)] = 0
print(test_user_correlation)


# In[133]:


test_user_correlation[test_user_correlation<0]=0
test_user_predicted_ratings = np.dot(test_user_correlation, test_beer_features.fillna(0))
test_user_predicted_ratings


# In[134]:


#Doing prediciton for the beer rated by the user
test_user_final_rating = np.multiply(test_user_predicted_ratings,dummy_test)


# In[135]:


test_user_final_rating.head()


# In[136]:


#Calculating the RMSE for only the beer rated by user. 
#For RMSE, normalising the rating to (1,5) range
from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = test_user_final_rating.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[137]:


test_ = test.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
)


# In[138]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[139]:


rmse = (sum(sum((test_ - y )**2))/total_non_nan)**0.5
print(rmse)


# In[140]:


#Using Item Similarity
test_beer_features = test.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
).T

mean = np.nanmean(test_beer_features, axis=1)
test_df_subtracted = (test_beer_features.T-mean).T

test_item_correlation = 1 - pairwise_distances(test_df_subtracted.fillna(0), metric='cosine')
test_item_correlation[np.isnan(test_item_correlation)] = 0
test_item_correlation[test_item_correlation<0]=0


# In[141]:


test_item_correlation.shape


# In[142]:


test_beer_features.shape


# In[143]:


test_item_predicted_ratings = (np.dot(test_item_correlation, test_beer_features.fillna(0))).T
test_item_final_rating = np.multiply(test_item_predicted_ratings,dummy_test)
test_item_final_rating.head()


# In[144]:


test_ = test.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
)


# In[145]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = test_item_final_rating.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))


test_ = test.pivot(
    index='review_profilename',
    columns='beer_beerid',
    values='review_overall'
)

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[146]:


rmse = (sum(sum((test_ - y )**2))/total_non_nan)**0.5
print(rmse)


# ### User similarity is better since the RMSE is lesser

# ### Q.  Give the names of the top 5 beers that you would recommend to the users 'cokes', 'genog' and 'giblet' using both the models.

# - The top 5 beers that you would recommend to the users 'cokes' using User based model

# In[242]:


userBasedPrediction = test_user_final_rating.reset_index()

coke_prediction = pd.DataFrame(
    userBasedPrediction.loc[userBasedPrediction['review_profilename'] == 'cokes']).T.reset_index()

coke_prediction.rename(columns={"beer_beerid": "Beer", 5530: "User_Rating"}, inplace= True)

coke_prediction_1 = pd.DataFrame(coke_prediction[~coke_prediction.User_Rating.isin([NaN])])
coke_prediction_2 = pd.DataFrame(coke_prediction_1[~coke_prediction_1.User_Rating.isin(['cokes'])])

coke_prediction_2.sort_values('User_Rating', axis = 0, ascending = False)[:5]


# - The top 5 beers that you would recommend to the users 'genog' using User based model

# In[243]:


coke_prediction = pd.DataFrame(
    userBasedPrediction.loc[userBasedPrediction['review_profilename'] == 'genog']).T.reset_index()
coke_prediction.rename(columns={"beer_beerid": "Beer", 6404: "User_Rating"}, inplace= True)

coke_prediction_1 = pd.DataFrame(coke_prediction[~coke_prediction.User_Rating.isin([NaN])])
coke_prediction_2 = pd.DataFrame(coke_prediction_1[~coke_prediction_1.User_Rating.isin(['genog'])])

coke_prediction_2.sort_values('User_Rating', axis = 0, ascending = False)[:5]


# - The top 5 beers that you would recommend to the users 'giblet' using User based model

# In[244]:


coke_prediction = pd.DataFrame(
    userBasedPrediction.loc[userBasedPrediction['review_profilename'] == 'giblet']).T.reset_index()
coke_prediction.rename(columns={"beer_beerid": "Beer", 6436: "User_Rating"}, inplace= True)

coke_prediction_1 = pd.DataFrame(coke_prediction[~coke_prediction.User_Rating.isin([NaN])])
coke_prediction_2 = pd.DataFrame(coke_prediction_1[~coke_prediction_1.User_Rating.isin(['giblet'])])

coke_prediction_2.sort_values('User_Rating', axis = 0, ascending = False)[:5]


# -The top 5 beers that you would recommend to the users 'cokes' using Item based model

# In[246]:


ItemBasedPrediction = test_item_final_rating.reset_index()

coke_prediction = pd.DataFrame(
    ItemBasedPrediction.loc[userBasedPrediction['review_profilename'] == 'cokes']).T.reset_index()

coke_prediction.rename(columns={"beer_beerid": "Beer", 5530: "User_Rating"}, inplace= True)

coke_prediction_1 = pd.DataFrame(coke_prediction[~coke_prediction.User_Rating.isin([NaN])])
coke_prediction_2 = pd.DataFrame(coke_prediction_1[~coke_prediction_1.User_Rating.isin(['cokes'])])

coke_prediction_2.sort_values('User_Rating', axis = 0, ascending = False)[:5]


# In[249]:


ItemBasedPrediction = test_item_final_rating.reset_index()

coke_prediction = pd.DataFrame(
    ItemBasedPrediction.loc[userBasedPrediction['review_profilename'] == 'genog']).T.reset_index()

coke_prediction.rename(columns={"beer_beerid": "Beer", 6404: "User_Rating"}, inplace= True)

coke_prediction_1 = pd.DataFrame(coke_prediction[~coke_prediction.User_Rating.isin([NaN])])
coke_prediction_2 = pd.DataFrame(coke_prediction_1[~coke_prediction_1.User_Rating.isin(['genog'])])

coke_prediction_2.sort_values('User_Rating', axis = 0, ascending = False)[:5]


# In[252]:


ItemBasedPrediction = test_item_final_rating.reset_index()

coke_prediction = pd.DataFrame(
    ItemBasedPrediction.loc[userBasedPrediction['review_profilename'] == 'giblet']).T.reset_index()

coke_prediction.rename(columns={"beer_beerid": "Beer", 6436: "User_Rating"}, inplace= True)

coke_prediction_1 = pd.DataFrame(coke_prediction[~coke_prediction.User_Rating.isin([NaN])])
coke_prediction_2 = pd.DataFrame(coke_prediction_1[~coke_prediction_1.User_Rating.isin(['giblet'])])

coke_prediction_2.sort_values('User_Rating', axis = 0, ascending = False)[:5]


# <h1><center>THE END</center></h1>

# <h1><center>Thank You..!</center></h1>

# In[ ]:




