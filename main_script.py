import os
import sys
from load_data import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''
Develop a Machine Learning model for a company to invest in real estate in california.
We have housing Data, try to predict the housing value from the available attributes 


'''


def load_housing_data(housing_path = "../"+HOUSING_PATH):
    
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()

#### Visualization Methods

print("Printing first elements:")
print(housing.head())
print()

## iloc prints data at particular index

print( housing.iloc[0] )

print("Printing all attribute / column info:")
print(housing.info())
print()

## Number of categories and how many
print("Printing Categories of Ocean Statistics:")
print( housing["ocean_proximity"].value_counts() )
print()

## Describe the statictics of the dataset
print("Printing Statistics:")
print(housing.describe() )
print()

## Plotting Histogram of all attributes

fig = housing.hist(bins=50, figsize =(12 , 12) )

#plt.savefig("histogram_dist_dataset")
plt.show()

def split_train_test( data, test_ratio):
    
    ### If we want to maintain the same split , we give it a seed
    #np.random.seed(42)
    
    shuffled_indices = np.random.permutation( len(data) )
    test_set_size = int( len(data)*test_ratio )
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    print("Created Test and Train Split at "+str(test_ratio)+" split" )
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

print("Length of Original Data: ", len(housing))
print("Length of new Training set: ", len(train_set))
print("Length of new Testing set: ", len(test_set))


### Same thing , but using sklearn

from sklearn.model_selection import train_test_split

train_set_sk, test_set_sk = train_test_split( housing, test_size=0.2, random_state=42)


'''
Sampling Bias:

Lets say That during development , we know that the median income is very import and highly correlated to 
housing value, Doing a normal sampling, might not give the best representative distribution for housing income

Soln: Perform Stratified sampling
Creating a new category for median_income to sample equally from each strate = Stratified Sampling
we will now categorize median income and then sample equally from each strate
'''
housing["income_cat"] = pd.cut( housing["median_income"], bins =[0., 1.5, 3.0, 4.5, 6., np.inf], labels = [1,2,3,4,5] )

### Notice how new attribute has been created
print(housing.iloc[0])
print(housing.head())

housing["income_cat"].hist()

### Stratified Sampling from scikit

from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit (n_splits=1, test_size=0.2 , random_state=42)

for train_index, test_index in strat_split.split(housing, housing["income_cat"] ):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
### Testing the stratified split

print("StratifiedSampling Random Split:" )
print( strat_test_set["income_cat"].value_counts()/ len(strat_test_set) )


### Testing the Original split
from sklearn.model_selection import train_test_split

train_set_sk, test_set_sk = train_test_split( housing, test_size=0.2, random_state=42)
print("Original Random Split:" )
print( test_set_sk["income_cat"].value_counts()/ len(test_set_sk) )


housing_vis = strat_train_set

housing_vis.plot(kind = 'scatter', x='longitude', y='latitude')

housing_vis.plot(kind = 'scatter', x='longitude', y='latitude', alpha=0.1)

housing_vis.plot(kind = 'scatter', x='longitude', y='latitude', alpha=0.4, s = housing['population']/100, label='population',
                 figsize=(10,7), c='median_house_value', cmap= plt.get_cmap('jet'), colorbar=True)


'''
Finding Correlation between every pair of attributes

In statistics, the Pearson correlation coefficient (PCC, pronounced /ˈpɪərsən/), also referred to as Pearson's r, 
the Pearson product-moment correlation coefficient (PPMCC) or the bivariate correlation,[1] is a measure of the 
linear correlation between two variables X and Y. According to the Cauchy–Schwarz inequality it has a value between +1 and −1,
 where 1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation.

'''

corr_matrix = housing_vis.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

#plt.matshow(corr_matrix)
#plt.show()

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);

plot_corr(housing_vis)


'''
Another way to check for correlation between attributes is to use Pandas’
scatter_matrix function, which plots every numerical attribute against every other
numerical attribute. Since there are now 11 numerical attributes, you would get 112
 =
121 plots, which would not fit on a page, so let’s just focus on a few promising
attributes that seem most correlated with the median housing value
'''

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix ( housing_vis[attributes] , figsize=(12,8) )

housing_vis.plot(kind= 'scatter', x='median_income',y= 'median_house_value', alpha=0.1)





housing_vis["rooms_per_household"] = housing_vis["total_rooms"]/housing_vis["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing_vis.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))


''' 
Machine Learning

making data readdy for ML
'''
## Preparing for Transformations


housing = strat_train_set.drop("median_house_value", axis =1)
housing_labels = strat_train_set["median_house_value"].copy()
# drop creates a copy and does not modify the original set

# Total bedrooms has missing values
#1- Get rid of the corresponding districts.
#2 -Get rid of the whole attribute.
#3 -Set the values to some value (zero, the mean, the median, etc.)

#housing.dropna(subset=["total_bedrooms"]) # option 1
#housing.drop("total_bedrooms", axis=1) # option 2

median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)


# Simple imputer can learn and fill NA, 
# But only works on numerical values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# This has only numbers
housing_num = housing.drop("ocean_proximity", axis=1)


# Simply computes - doesn't transform
imputer.fit(housing_num)

print(imputer.statistics_)

X= imputer.transform(housing_num)
# This is a numpy array

housing_tr = pd.DataFrame(X, columns=housing_num.columns)






