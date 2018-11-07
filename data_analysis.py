# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:36:26 2018

@author: Koffi Moïse AGBENYA

Data Analysis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'

df = pd.read_csv(path)
print("Data successfully loaded!")
#print(df.head(3))   check if all is done!
#How to choose the right vizualisation method ?
#It's important to understand what type of variable we are dealing with
#print(df.dtypes)
#print(df.corr())  to check the correlation between variables with type int64 or float64
#Here we're checking the correlation between bore, stroke, compression-ratio and horsepower

cor1 = df[['bore','stroke','compression-ratio','horsepower']]
print(cor1.corr())
#Continuous numerical variables are variables that may contain any value within some range.
#Continuous numerical variables can have the type "int64" or "float64". A great way to
#visualize these variables is by using scatterplots with fitted lines.

#Here we are trying to understand the relationship between an individual variable and the price
#We can do this by using "regplot", which plots the scatterplot plus the fitted regression line
#for the data.

# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
#As the engine-size goes up, the price goes up: this indicates a positive direct correlation
#between these two variables. Engine size seems like a pretty good predictor of price since
#the regression line is almost a perfect diagonal line.

#Now we can examine correlation between those two variables to check the result
print(df[["engine-size", "price"]].corr())
# correlation(engine-size, price) = 0.872335. In resume it's a positive relationship

#Highway-mpg is also a potential predictor variable of price
sns.regplot(x="highway-mpg", y="price", data=df)
#As the highway-mpg goes up, the price goes down: this indicates an inverse/ negative 
#relationship between these two variables. Highway mpg could potentially be a predictor of price
#Now we can check the correlation between those two variables
print(df[["highway-mpg", "price"]].corr())
# The result is -0.704692


#Check of categorical variables
#These are variables that describe a 'characteristic' of a data unit, and are selected from
#a small group of categories. The categorical variables can have the type "object" or "int64".
#A good way to visualize categorical variables is by using boxplots.
#Let's look at the relationship between "body-style" and "price".
sns.boxplot(x="body-style", y="price", data=df)

#We see that the distributions of price between the different body-style categories have
#a significant overlap, and so body-style would not be a good predictor of price.
#Let's examine engine "engine-location" and "price" :
sns.boxplot(x="engine-location", y="price", data=df)
#Here we see that the distribution of price between these two engine-location categories,
#front and rear, are distinct enough to take engine-location as a potential good predictor
#of price.

#Let's examine "drive-wheels" and "price".
# drive-wheels
sns.boxplot(x="drive-wheels", y="price", data=df)
#Here we see that the distribution of price between the different drive-wheels categories
#differs; as such drive-wheels could potentially be a predictor of price.


#Descriptive statistical analysis
#The describe function automatically computes basic statistics for all continuous variables.
#Any NaN values are automatically skipped in these statistics. We are including object variable
print(df.describe())
print(df.describe(include=['object']))

#Value-counts is a good way of understanding how many units of each
#characteristic/variable we have.
#the method "value_counts" only works on Pandas series, not Pandas Dataframes.
df['drive-wheels'].value_counts()
#Apply to the column drive-wheels we obtain fwd    118  rwd     75  4wd      8
#We can convert the series to a dataframe as follows :
df['drive-wheels'].value_counts().to_frame()

# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts"
#and rename the column  'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
print(drive_wheels_counts)
#Now let's rename the index to 'drive-wheels':)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

#Let's repeat the above process for the variable 'engine-location'.
# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))
#Examining the value counts of the engine location would not be a good predictor variable
#for the price. This is because we only have three cars with a rear engine and 198 with an 
#engine in the front, this result is skewed. Thus, we are not able to draw any conclusions 
#about the engine location.



#The "groupby" method groups data by different categories. The data is 
#grouped based on one or several variables and analysis is performed on the individual groups.
#Let's group by the variable "drive-wheels"
df['drive-wheels'].unique()
#We see that there are 3 different categories of drive wheels.
#Let's select the columns 'drive-wheels','body-style' and 'price' , then assign
#it to the variable "df_group_one".
df_group_one=df[['drive-wheels','body-style','price']]
# grouping results
#Now We compute the average, to search which type of drive-wheels are most valuable
df_group_one=df_group_one.groupby(['drive-wheels'],as_index= False).mean()
print(df_group_one)

#From our data, it seems rear-wheel drive vehicles are, on average, the most expensive, 
#while 4-wheel and front-wheel are approximately the same in price.
#We can also group with multiple variables. For example, let's group by both
#'drive-wheels' and 'body-style'. This groups the dataframe by the unique combinations 
#'drive-wheels' and 'body-style'
# grouping results
df_gptest=df[['drive-wheels','body-style','price']]
grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index= False).mean()
print(grouped_test1)

#This grouped data is much easier to visualize when it is made into a pivot table. 
#A pivot table is like an Excel spreadsheet, with one variable along the column and 
#another along the row. We can convert the dataframe to a pivot table using the method 
#"pivot " to create a pivot table from the groups.
#In this case, we will leave the drive-wheel variable as the rows of the table, and pivot
#body-style to become the columns of the table:

grouped_pivot=grouped_test1.pivot(index='drive-wheels',columns='body-style')
print(grouped_pivot)
#Often, we won't have data for some of the pivot cells. We can fill these missing cells 
#with the value 0, but any other value could potentially be used as well
grouped_pivot=grouped_pivot.fillna(0) #fill missing values with 0 !!!It's not a good way
print(grouped_pivot)

#Lets use a heatmap to visualize the relationship between body style and price
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

#The heatmap plots the target variable (price) proportional to colour with respect 
#to the variables 'drive-wheel' and 'body-style' in the vertical and horizontal axis 
#respectively. This allows us to visualize how the price is related to 'drive-wheel' 
#and 'body-style', 
#The default labels convey no useful information to us. Let's change that:

fig, ax=plt.subplots()
im=ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels=grouped_pivot.columns.levels[1]
col_labels=grouped_pivot.index
#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0])+0.5, minor=False)
#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

#he main question we want to answer, is "What are the main characteristics 
#which have the most impact on the car price?".

#To get a better measure of the important characteristics, we look at the correlation of 
#these variables with the car price, in other words: how is the car price dependent on 
#this variable?


### Wheel-base vs Price
#Let's calculate the  Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'. 
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
#Since the p-value is < 0.001, the correlation between wheel-base and price is 
#statistically significant, although the linear relationship isn't extremely strong (~0.585)

### Horse-power vs price
#Let's calculate the Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

#Since the p-value is < 0.001, the correlation between horsepower and price is statistically
#significant, and the linear relationship is quite strong (~0.809, close to 1)

#Length vs Price
#Let's calculate the Pearson Correlation Coefficient and P-value of 'length' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#Since the p-value is < 0.001, the correlation between length and price 
#is statistically significant, and the linear relationship is moderately strong (~0.691).


#Width vs Price
#Let's calculate the Pearson Correlation Coefficient and P-value of 'width' and 'price':
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )
#Since the p-value is < 0.001, the correlation between width and price 
#is statistically significant, and the linear relationship is quite strong (~0.751).

#Curb-weight vs Price
#Let's calculate the Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price':
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#Since the p-value is < 0.001, the correlation between curb-weight and price is statistically 
#significant, and the linear relationship is quite strong (~0.834).

#Engine-size vs Price
#Let's calculate the Pearson Correlation Coefficient and P-value of 'engine-size' and 'price':
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#Since the p-value is < 0.001, the correlation between engine-size and price is statistically 
#significant, and the linear relationship is very strong (~0.872).

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

#Since the p-value is < 0.001, the correlation between bore and price is statistically 
#significant, but the linear relationship is only moderate (~0.521).


# 'City-mpg' and 'Highway-mpg':

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
#Since the p-value is < 0.001, the correlation between city-mpg and price is statistically 
#significant, and the coefficient of ~ -0.687 shows that the relationship is negative 
#and moderately strong.

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 
#Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically 
#significant, and the coefficient of ~ -0.705 shows that the relationship is negative 
#and moderately strong.


#Analysis of Variance
grouped_test2=df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
grouped_test2.head(2)

#We can obtain the values of the method group using the method "get_group".
grouped_test2.get_group('4wd')['price']
#we can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.
# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val) 

#This is a great result, with a large F test score showing a strong correlation and 
#a P value of almost 0 implying almost certain statistical significance. But does 
#this mean all three tested groups are all this highly correlated?
#Separately: fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )

#Let's examine the other groups

#4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val) 

#4wd and fwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)   


#We now have a better idea of what our data looks like and which variables are important to 
#take into account when predicting the car price. We have narrowed it down to the following 
#variables:
