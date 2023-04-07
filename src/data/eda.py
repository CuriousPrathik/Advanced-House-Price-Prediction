# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns',None)


# load the data

# Set the path to the raw data folder
raw_data_path = 'C:\\Users\\prath\\Advanced-House-Price-Prediction\\data\\raw\\'

# Load the train.csv file into a pandas DataFrame
df_train = pd.read_csv(raw_data_path + 'train.csv')


# Here we will check the percentage of NaN values present in each feature

# 1 -step make the list of features which has missing values
features_with_na = [features for features in df_train.columns if df_train[features].isnull().sum() >1]

# 2- step print the feature name and the percentage of missing values    

for feature in features_with_na:
    print(feature, np.round(df_train[feature].isnull().mean(),4), '% missing values')


# list of numerical variables
numerical_features = [feature for feature in df_train.columns if df_train[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df_train[numerical_features].head()


# list of variable that contain year information

year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature


# Check whether there is a relation between year the house is sold and the sales price

df_train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# list of discrete numerical features

discrete_feature=[feature for feature in numerical_features if len(df_train[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))

# visualise the numerical variables
df_train[discrete_feature].head()


# Realtion between descrete features and Sale PRice

for feature in discrete_feature:
    data=df_train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# list of continous numerical features

continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))

# visualise the numerical variables
df_train[continuous_feature].head()


# Analyse by creating histograms to understand the distribution

for feature in continuous_feature:
    data=df_train.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# Using log transformation to reduce the skewness of data

for feature in continuous_feature:
    data=df_train.copy()
    # We used 'If 0 in dataset[feature].unique()' to avoid log(0) condition. Because log(0) is not defined
    if 0 in data[feature].unique():
        pass
    else:
        # taking values that does not have 0
        # transforming both feature and saleprice
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        # Scatter plot
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()


# create box plots

for feature in continuous_feature:
    data=df_train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# list of categorical features

categorical_features=[feature for feature in df_train.columns if data[feature].dtypes=='O']
print("Categorical feature Count {}".format(len(categorical_features)))

# visualise the categorical variables
df_train[categorical_features].head()


# unique categories/Cardinality of Categorical Variables

for feature in categorical_features:
    print(f'The feature is {feature} and number of categories are {len(df_train[feature].unique())}')


# relationship between categorical variable and SalesPrice

for feature in categorical_features:
    data=df_train.copy()
    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()