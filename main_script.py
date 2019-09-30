import os
import sys
from load_data import *
import pandas as pd
import matplotlib.pyplot as plt

def load_housing_data(housing_path = "../"+HOUSING_PATH):
    
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()

#### Visualization Methods

print("Printing first elements:")
print(housing.head())
print()

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

plt.savefig("histogram_dist_dataset")
plt.show()