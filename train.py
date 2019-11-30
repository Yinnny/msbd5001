import numpy as np
import pandas as pd
from collections import Counter
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('/Users/yin/5001/msbd5001-fall2019/train.csv')
test = pd.read_csv('/Users/yin/5001/msbd5001-fall2019/test.csv')
train['all_genres']=train['genres'].str.split(',')
train['num_genres']=train['all_genres'].apply(lambda x:len(x))
list_of_genres=list(train['all_genres'].apply(lambda x: [i for i in x] ).values)
top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common()]

for g in top_genres:
    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)

test['all_genres']=test['genres'].str.split(',')
test['num_genres']=test['all_genres'].apply(lambda x:len(x))
list_of_genres1=list(test['all_genres'].apply(lambda x: [i for i in x] ).values)
top_genres1 = [m[0] for m in Counter([i for j in list_of_genres1 for i in j]).most_common()]

for g in top_genres1:
    test['genre_' + g] = test['all_genres'].apply(lambda x: 1 if g in x else 0)

train['all_categories'] = train['categories'].str.split(',')
train['num_categories'] = train['all_categories'].apply(lambda x: len(x))
list_of_categories=list(train['all_categories'].apply(lambda x: [i for i in x] ).values)
top_categories = [m[0] for m in Counter([i for j in list_of_categories for i in j]).most_common()]

for g in top_categories:
    train['categories_' + g] = train['all_categories'].apply(lambda x: 1 if g in x else 0)

test['all_categories'] = test['categories'].str.split(',')
test['num_categories'] = test['all_categories'].apply(lambda x: len(x))
list_of_categories1=list(test['all_categories'].apply(lambda x: [i for i in x] ).values)
top_categories1 = [m[0] for m in Counter([i for j in list_of_categories1 for i in j]).most_common()]

for g in top_categories1:
    test['categories_' + g] = test['all_categories'].apply(lambda x: 1 if g in x else 0)

train['all_tags']=train['tags'].str.split(',')
train['num_tags']=train['all_tags'].apply(lambda x:len(x))
list_of_tags=list(train['all_tags'].apply(lambda x: [i for i in x] ).values)
top_tags = [m[0] for m in Counter([i for j in list_of_tags for i in j]).most_common()]

for g in top_tags:
    train['tags_' + g] = train['all_tags'].apply(lambda x: 1 if g in x else 0)

test['all_tags']=test['tags'].str.split(',')
test['num_tags']=test['all_tags'].apply(lambda x:len(x))
list_of_tags1=list(test['all_tags'].apply(lambda x: [i for i in x] ).values)
top_tags1 = [m[0] for m in Counter([i for j in list_of_tags1 for i in j]).most_common()]

for g in top_tags1:
    test['tags_' + g] = test['all_tags'].apply(lambda x: 1 if g in x else 0)

train['purchase_date'] = pd.to_datetime(train['purchase_date'])
train['release_date'] = pd.to_datetime(train['release_date'])
train['purchase_year'] = train['purchase_date'].apply(lambda x: x.year)
train['purchase_month'] = train['purchase_date'].apply(lambda x: x.month)
train['purchase_day'] = train['purchase_date'].apply(lambda x: x.day)
train['release_year'] = train['release_date'].apply(lambda x: x.year)
train['release_month'] = train['release_date'].apply(lambda x: x.month)
train['release_day'] = train['release_date'].apply(lambda x: x.day)
train['days']=(train['purchase_date']-train['release_date']).map(lambda x:x.days)

test['purchase_date'] = pd.to_datetime(test['purchase_date'])
test['release_date'] = pd.to_datetime(test['release_date'])
test['purchase_year'] = test['purchase_date'].apply(lambda x: x.year)
test['purchase_month'] = test['purchase_date'].apply(lambda x: x.month)
test['purchase_day'] = test['purchase_date'].apply(lambda x: x.day)
test['release_year'] = test['release_date'].apply(lambda x: x.year)
test['release_month'] = test['release_date'].apply(lambda x: x.month)
test['release_day'] = test['release_date'].apply(lambda x: x.day)
test['days']=(test['purchase_date']-test['release_date']).map(lambda x:x.days)

#fill the missing values
train.loc[(train.purchase_year.isnull()), 'purchase_year'] = train.purchase_year.dropna().mean()
train.loc[(train.purchase_month.isnull()), 'purchase_month'] = train.purchase_month.dropna().mean()
train.loc[(train.purchase_day.isnull()), 'purchase_day'] = train.purchase_day.dropna().mean()
train.loc[(train.days.isnull()), 'days'] = train.days.dropna().mean()
train.loc[(train.total_positive_reviews.isnull()), 'total_positive_reviews'] = train.total_positive_reviews.dropna().mean()
train.loc[(train.total_negative_reviews.isnull()), 'total_negative_reviews'] = train.total_negative_reviews.dropna().mean()

test.loc[(test.purchase_year.isnull()), 'purchase_year'] = test.purchase_year.dropna().mean()
test.loc[(test.purchase_month.isnull()), 'purchase_month'] = test.purchase_month.dropna().mean()
test.loc[(test.purchase_day.isnull()), 'purchase_day'] = test.purchase_day.dropna().mean()
test.loc[(train.days.isnull()), 'days'] = test.days.dropna().mean()
test.loc[(test.total_positive_reviews.isnull()), 'total_positive_reviews'] = test.total_positive_reviews.dropna().mean()
test.loc[(test.total_negative_reviews.isnull()), 'total_negative_reviews'] = test.total_negative_reviews.dropna().mean()


train["positive_rate"]=train["total_positive_reviews"]/(train["total_positive_reviews"]+train["total_negative_reviews"])
test["positive_rate"]=test["total_positive_reviews"]/(test["total_positive_reviews"]+test["total_negative_reviews"])
train['play']=np.where(train['playtime_forever']==0,0,1)
y=train['playtime_forever']
y2=train['play']

data=train.copy()
X=data.drop(['genres','purchase_date','categories','tags','release_date','all_genres', 'all_categories', 'all_tags'], axis=1)
test_data=test.copy()
test_data=test_data.drop(['genres','purchase_date','categories','tags','release_date','all_genres', 'all_categories', 'all_tags'],  axis=1)


data1=X.copy()
data1=data1.drop(['playtime_forever'],axis=1)
data1=data1.drop(['play'],axis=1)
data2=test_data.copy()
x_train = data1
x_test = data2
feature=data1.columns & data2.columns
x_train = x_train.loc[:, feature]
x_test = x_test.loc[:, feature]

less_feature = []
most_feature = []
features = x_train.columns[30:]

for i in features:
    if ((np.sum(data[i] == 1) <= 20)):
        less_feature.append(i)


x_train = x_train.drop(less_feature, axis=1)
x_test = x_test.drop(less_feature, axis=1)

#x_train=x_train.loc[:,data2.columns]
print(x_train.shape)
print(x_test.shape)
x_train=x_train.fillna(0.0)
x_test=x_test.fillna(0.0)
model1 = RandomForestClassifier(n_estimators= 75, max_depth=60, min_samples_leaf=20, max_features=60)
model=xgb.XGBRegressor(learning_rate= 0.01, n_estimators= 240, max_depth=8, min_child_weight= 4, seed= 0,
                subsample= 0.8, colsample_bytree=0.6, gamma= 0.1, reg_alpha=3, reg_lambda= 0.05)
model1.fit(x_train,y2)
model.fit(x_train,y)
play=model1.predict(x_test)
feature_importance = model.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
print('重要特征：', x_train.columns)
print('每个重要特征的重要性：', feature_importance)

x_train = x_train[x_train.columns[sorted_idx][::-1][:-20]]
x_test = x_test[x_test.columns[sorted_idx][::-1][:-20]]
model=xgb.XGBRegressor(learning_rate= 0.01, n_estimators= 240, max_depth=8, min_child_weight= 4, seed= 0,
                subsample= 0.8, colsample_bytree=0.6, gamma= 0.1, reg_alpha=3, reg_lambda= 0.05)
model.fit(x_train,y)


x_test['playtime_forever']=model.predict(x_test)
x_test['play']=play
x_test['playtime_forever'] = np.where(x_test['play']==0,0,x_test['playtime_forever'])

test['playtime_forever']=x_test['playtime_forever']
test['playtime_forever']=test['playtime_forever'].apply(lambda x: 0 if x<0 else x)
df=test.loc[:,['id','playtime_forever']]
df.to_csv('/Users/yin/5001/msbd5001-fall2019/output1010.csv', index=False)

print("number of features"+str(x_train.shape[1]))
print('done!')



