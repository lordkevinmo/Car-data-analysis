# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

@author: Koffi_AGBENYA
"""
#panda, numPy libraries import
import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

#url for data
path="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

#structure of headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

#read data to dataframe
df = pd.read_csv(path, header=None)
#Just to verify if all going well
print("Done")
#Associate headers to the dataframe
df.columns = headers
#print 5 frirst row for analyse trhe constitution of the data
#print(df.head())
#In column normalize-losses, we hame missing data, represent by ?
# we will replace it to NaN
df.replace("?", np.nan, inplace=True)
#print(df.head())
#Evaluating of missing data
missing_value = df.isnull()
#print(missing_value)
#With this analysis we found many true in the dataframe => there missing value
# Now I'm going to count missing values in each column
for column in missing_value.columns.values.tolist():
    print(column)
    print(missing_value[column].value_counts())
    print("")
#Many column has already good data
# Deal with missing data
# Here, we have two methods : drop data or replace data
# In our dataset, none of the columns are empty enough to drop whole column. We'll replace
# data until.
# We are using the mean value to replace NaN in each column.
#average of first missing values column (normalized-losses)
avg_1 = df["normalized-losses"].astype("float").mean(axis = 0)
df["normalized-losses"].replace(np.nan, avg_1, inplace = True)
#average of column 'bore'
avg_2 = df["bore"].astype("float").mean(axis=0)
df["bore"].replace(np.nan, avg_2, inplace = True)

#average of ..
avg_3 = df["stroke"].astype("float").mean(axis=0)
df["stroke"].replace(np.nan, avg_3, inplace = True)

# horsepower average
avg_4=df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_4, inplace= True)

# peak-rpm average
avg_5=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_5, inplace= True)

df['num-of-doors'].value_counts()
# calculate for us the most common type 
df['num-of-doors'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace = True)
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace = True)

# reset index, because we droped two rows
df.reset_index(drop = True, inplace = True)

#lets display the first 5 row
#print(df.head())

#lets lists the data format of each column
#print(df.dtypes)

#Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print("Data format Done!")

#verification
#print(df.dtypes)
#parfait

""" 
    Data Standardization

"""
# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

"""
   Data Normalisation
"""

# replace (origianl value) by (original value)/(maximum value) for length, width, height, ..
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max() 
# show the scaled columns
#print(df[["length","width","height"]].head())
print("Data standardization and normalisation done!")

"""
    Data Binning
"""
# Convert data to correct form
df["horsepower"]=df["horsepower"].astype(float, copy=True)

# We would like four bins of equal size bandwidth
binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4

# We build a bin array, with a minimum value to a maximum value, with bandwidth
# calculated above. The bins will be values used to determine when one bin ends
# and another begins.
bins = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
#print(bins)

#Set of group-name
group_names = ['Low', 'Medium', 'High']

#We apply the function "cut" the determine what each value of "df['horsepower']" belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,include_lowest=True )
# Check of binning
#print(df[['horsepower','horsepower-binned']].head(20))
# Yeah! we successfully narrow the intervals from 57 to 3!

# Binds visualization
a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

#Dummies variable
#We see the column "fuel-type" has two unique values, "gas" or "diesel".
# Regression doesn't understand words, only numbers. To use this attribute in regression
# analysis, we convert "fuel-type" into indicator variables.
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
#print(dummy_variable_1.head())
#change column name for clarity
dummy_variable_1.rename(columns={'fuel-type-gas':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
#print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
#print(df.head())

# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
#print(dummy_variable_2.head())

#merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)

#Save data in new csv file
df.to_csv('clean_df.csv')