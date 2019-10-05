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






