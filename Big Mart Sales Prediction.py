# %% [markdown]
# -----------------------------------------------------------------------------
# 1) Problem Statement 
# -----------------------------------------------------------------------------
# 
# 
# The Data Scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.  
# 
# Business Goal : Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.

# %% [markdown]
# analyse -
# 1) property of prd - weight , fat content , visibility , type , MRP
# 2) store - size , location type , outlet type 

# -----------------------------------------------------------------------------
# 2) Loading Packages 
# -----------------------------------------------------------------------------
import numpy as np # linear algebra
import pandas as pd # data processing
import math
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from xgboost import XGBRegressor


# Ignore warnings ;)
import warnings
warnings.simplefilter("ignore")

import pickle

# set seed for reproductibility
np.random.seed(0)

# %% [markdown]
# -----------------------------------------------------------------------------
# 3) Data Loading
# -----------------------------------------------------------------------------

# %%
import pandas as pd
train=pd.read_csv(r'C:\Users\rahimn\Downloads\train_v9rqX0R.csv')
test=pd.read_csv(r'C:\Users\rahimn\Downloads\test_AbJTz2l.csv')

# %%
# Preserve identifiers from TEST data for submission
test_ids = test.loc[:, ["Item_Identifier", "Outlet_Identifier"]].reset_index(drop=True)

# %% [markdown]
# merging the train and test data , to simplify the EDA , Feature engineering steps 

# %%
train["data_type"] = "train"
test["data_type"]  = "test"

df = pd.concat([train, test], ignore_index=True)

# %%
# -----------------------------------------------------------------------------
# 4) Data Structure and Content
# -----------------------------------------------------------------------------
df["data_type"].value_counts()


# %%
df.info()
#weight & outlet size has missing vals

# %%
#include='all' as we need to see both categorical as well as numberic attributes of prd , store
df.describe(include='all')

# %%
# how many store/its data 
df['Outlet_Identifier'].value_counts() 

# %%
df.isnull().sum()
# how many missing vals 
# Item_Weight - is numeric feature
# Outlet_Size - is categorical

# %% [markdown]
# # Exploratory Data Analysis - EDA

# %%
num_cols=df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
#for numberic vairables
sns.pairplot(df)
plt.show()

# %% [markdown]
# - sales of item weight range from 5 Kg to 20 Kg.
# - Item_Visibility feature is right skewed.
# - There are more products in the range of >100 MRP has more sale
# - We can observe that a lots of stores have been established in the years 1985, 1998 etc... and there was no store establishment between 1990 and 1995.
# - Item_Outlet_Sales feature is right skewed. 

# %%
#correlation_matrix
corr_m = df.corr(numeric_only=True) 
f, ax = plt.subplots(figsize =(7,6)) 
sns.heatmap(corr_m,annot=True, cmap ="YlGnBu", linewidths = 0.1) 
#Item_mrp showing more effect

# %%
### OUTLET wise sales ###
df.groupby(["Outlet_Identifier"])["Item_Outlet_Sales"].agg("mean").sort_values(ascending=False)

# %%
df.groupby(["Outlet_Identifier","Outlet_Size"])["Item_Outlet_Sales"].agg("mean").sort_values(ascending=False)
# small & Medium doing good

# %%
#see how item type related to outlet
df.groupby(["Outlet_Identifier","Item_Type"])["Item_Outlet_Sales"].agg("mean").sort_values().groupby("Outlet_Identifier").tail(1)

# %%
# supermarket dominant 
df.groupby(["Outlet_Identifier","Outlet_Type"])["Item_Outlet_Sales"].agg("mean").sort_values(ascending=False)

# %% [markdown]
# #### Visualising Categorical Variables

# %%
plt.figure(figsize=(20, 12))
plt.subplot(2, 3, 1)
sns.boxplot(x="Outlet_Type", y="Item_Outlet_Sales", data=df)
plt.subplot(2, 3, 2)
sns.boxplot(x="Outlet_Location_Type", y="Item_Outlet_Sales", data=df)
plt.subplot(2, 3, 3)
sns.boxplot(x="Outlet_Size", y="Item_Outlet_Sales", data=df)
plt.subplot(2, 3, 4)
sns.boxplot(x="Outlet_Identifier", y="Item_Outlet_Sales", data=df)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



# %%
plt.figure(figsize=(40, 12))
plt.subplot(2, 1, 1)
sns.boxplot(x="Item_Type", y="Item_Outlet_Sales", data=df)
plt.subplot(2, 1, 2)
sns.boxplot(x="Item_Fat_Content", y="Item_Outlet_Sales", data=df)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# -----------------------------------------------------------------------------
# 5) Exploratory Data Analysis - EDA ; Preprocessing + Feature Engineering
# -----------------------------------------------------------------------------
# # Handling missing values
# #oulet_size impute by mode
# #weight imputed by mean

# %%
#handiling missing values
df.isnull().sum()

# %%
#wt replace by mean wt
df["Item_Weight"].fillna(df["Item_Weight"].mean(),inplace=True)

# %%
miss_values = df['Outlet_Size'].isnull()
miss_values 


# %%
# getting mode for missing values based on outlet typ
mode_of_Outlet_size = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

# %%
df.loc[miss_values, 'Outlet_Size'] = df.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

# %%
df.isnull().sum()

# %%
import seaborn as sns

sns.catplot(x='Outlet_Location_Type',kind='count',data=df,aspect=3)

# %% [markdown]
# # Feature Engineering 
# 
# 

# %%
df["data_type"].value_counts()

# %%
df_copy=df.copy()

# %% [markdown]
# # Encoding Categorical Variables

# %%
sns.catplot(x='Item_Fat_Content',kind='count',data=df,aspect=3)

# %%
df['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'},inplace = True)

# %%
sns.catplot(x='Item_Fat_Content',kind='count',data=df,aspect=3)

# %% [markdown]
# #Item_Identifier also has multiple values which can be grouped

# %%
df['Item_Identifier_Categories'] =df['Item_Identifier'].replace({'^FD[A-Z]*[0-9]*':'FD','^DR[A-Z]*[0-9]*':'DR','^NC[A-Z]*[0-9]*':'NC'},regex = True)

# %%
df=df.drop(columns=['Item_Identifier','Outlet_Establishment_Year'],axis=1)

# %%
sns.catplot(x='Item_Identifier_Categories',kind='count',data=df,aspect=3)

# %% [markdown]
# #Outlet_Location_Type label to number 1,2,3

# %%
df['Outlet_Location_Type'] = df['Outlet_Location_Type'].str[-1:].astype(int)

# %%
df.head()

# %%
encoder = LabelEncoder()
ordinal_features = ['Item_Fat_Content', 'Outlet_Type', 'Outlet_Location_Type']

for feature in ordinal_features:
    df[feature] = encoder.fit_transform(df[feature])

# %%
df.head()

# %%
df['Outlet_Size'] =df['Outlet_Size'].map({'Small'  : 1,'Medium' : 2,'High'   : 3}).astype(int)

# %%

# One-hot encoding (nominal)
cat_cols = ['Item_Type', 'Item_Identifier_Categories', 'Outlet_Identifier']

df2 = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int ,dummy_na=True)


# %%
df2.head()

# %% [markdown]
# #Separate train file (with labels)
# #Separate test file (no labels)
# #Validation split from training data only
# #Final prediction on X_test

# %%
X = df2.drop(columns='Item_Outlet_Sales', axis=1)
y = df2['Item_Outlet_Sales']


# %%
train_mask = X["data_type"] == "train"
test_mask  = X["data_type"] == "test"


# %%
X_train = X.loc[train_mask].drop(columns=["data_type"])
y_train = y.loc[train_mask]

X_test  = X.loc[test_mask].drop(columns=["data_type"])


# %% [markdown]
# -----------------------------------------------------------------------------
# 6) Modeling & Evaluation
# -----------------------------------------------------------------------------
# %% [markdown]
# -----------------------------------------------------------------------------
# 1) Linear Regression
# -----------------------------------------------------------------------------
# %%
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

lin_reg= LinearRegression()
lin_reg.fit(X_tr, y_tr)

val_pred = lin_reg.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
r2   = r2_score(y_val, val_pred)
                
print("Validation RMSE:", rmse)
print("R2:", r2)

# %% [markdown]
# #Split training data → train + validation
# #Fit on train split
# #Evaluate on validation split
# #Refit on full training data
# #Predict on test data

# %% [markdown]
# #refit on full training data, your final model as train uses only 80% of available data 
# #Validation score → from (X_val, y_val)
# #Test predictions → after refitting on full training data

# %%
lin_reg.fit(X_train, y_train)

# %%
test_pred = lin_reg.predict(X_test)

# %%
results = pd.DataFrame({'Method':['Linear Regression'],'RMSE': [rmse] , 'r2': [r2]})
results = results[['Method', 'RMSE', 'r2']]
results

# %% [markdown] 
# -----------------------------------------------------------------------------
# 2) Random Forest
# -----------------------------------------------------------------------------
# #Regression seems to have low score , lets do Random Forest to check 

# %%
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ridge = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=False)),  # safe for sparse or wide data
    ("model", Ridge(alpha=10.0, random_state=42))
])

ridge.fit(X_tr, y_tr)

val_pred = ridge.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
r2   = r2_score(y_val, val_pred)
                
print("Validation RMSE:", rmse)
print("R2:", r2)

# Refit on FULL training data
ridge.fit(X_train, y_train)

# Predict on test
ridge_test_pred = ridge.predict(X_test)

# %%
tempResults = pd.DataFrame({'Method':['Random Forest'], 'RMSE': [rmse],'r2': [r2] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'r2']]
results

# %% [markdown] 
# -----------------------------------------------------------------------------
# 3) LASSO Regression
# -----------------------------------------------------------------------------
# %%

from sklearn.linear_model import Lasso

lasso = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=False)),
    ("model", Lasso(alpha=0.001, max_iter=10000, random_state=42))
])


lasso.fit(X_tr, y_tr)

val_pred = lasso.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
r2   = r2_score(y_val, val_pred)
                
print("Validation RMSE:", rmse)
print("R2:", r2)

# Refit on FULL training data
lasso.fit(X_train, y_train)

# Predict on test
ridge_test_pred = lasso.predict(X_test)

# %%
tempResults = pd.DataFrame({'Method':['Lasso Regression'], 'RMSE': [rmse],'r2': [r2] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'r2']]
results

# %% [markdown]
# -----------------------------------------------------------------------------
# 4) XGBoost
# -----------------------------------------------------------------------------
# %%
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

val_pred = xgb.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
r2   = r2_score(y_val, val_pred)
                
print("Validation RMSE:", rmse)
print("R2:", r2)

# Refit on FULL training data
xgb.fit(X_train, y_train)

# Predict on test
xgb_pred = xgb.predict(X_test)

# %%
tempResults = pd.DataFrame({'Method':['XGBRegressor'], 'RMSE': [rmse],'r2': [r2] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'r2']]
results

# %% [markdown]
# #Linear Regression was selected as the final model as it achieved performance comparable to more complex models (Random Forest, Lasso) while offering greater interpretability, stability, and ease of deployment. The marginal differences in RMSE and R² were not statistically significant
# # Linear Regression
test_pred = np.maximum(0, test_pred)
#The negative values appear because we used unconstrained regression models. Linear and regularized regressions do not enforce business constraints such as non‑negative sales, so for some low‑signal item–outlet combinations the model extrapolates slightly below zero. This is a known and expected behavior of linear models
# %%
submission = pd.DataFrame({
    "Item_Identifier": test_ids["Item_Identifier"],
    "Outlet_Identifier": test_ids["Outlet_Identifier"],
    "Item_Outlet_Sales": test_pred   # this is linear model pred variable
})

submission.to_csv("submission.csv", index=False)
submission.head()
