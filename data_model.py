# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 01:09:27 2018

@author: Koffi Moïse AGBENYA

Model Development

 we will develop several models that will predict the price of the car using the
 variables or features. This is just an estimate but should give us an objective
 idea of how much the car should cost.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
# path of data 
path = 'https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)
#df.head() successfuly done!
### 1. Linear Regression and Multiple Linear Regression 
#Simple Linear Regression is a method to help us understand the relationship between two variables:
#The result of Linear Regression is a linear function that predicts the response 
#(dependent) variable as a function of the predictor (independent) variable.

#LinearRegression object creation
lm = LinearRegression()
#How could Highway-mpg help us predict car price?
#For this example, we want to look at how highway-mpg can help us predict car price. 
#Using simple linear regression, we will create a linear function with "highway-mpg" 
#as the predictor variable and the "price" as the response variable.
X = df[['highway-mpg']]
Y = df['price']
#I fit the l model using highway-mpg
lm.fit(X,Y)
#We can output a prediction
Yhat=lm.predict(X)
print(Yhat[0:5])
#Value of intercept
print(lm.intercept_)
#value of slope
print(lm.coef_)

#Another object
lr = LinearRegression()
#Train the model using 'engine-size' as the independent variable and 'price' as the dependent variable
lr.fit(df[['engine-size']],df['price'])
Yh = lr.predict(df[['engine-size']])
lr.intercept_
lr.coef_
#the equation of the predicted line is Price=38423.31-821.733*engine-size

#MLR
# Multiple Linear Regression is very similar to Simple Linear Regression, but this method is used 
#to explain the relationship between one continuous response (dependent) variable and two or more 
#predictor (independent) variables.
# From the previous analysis, we know that other good predictors of price could be: Horsepower, 
#Curb-weight, Engine-size and Highway-mp
#Let's develop a model using these variables as the predictor variables.
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#Fit the linear model using the four above-mentioned variables.
lm.fit(Z, df['price'])
lm.intercept_ #value of intercept
lm.coef_  #value of coefficient b1, b2, b3, b4
#we should get a final linear function with the structure: Yhat=a+b1X1+b2X2+b3X3+b4X4
#Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 
#81.95906216 x engine-size + 33.58258185 x highway-mpg

# Model evaluation using visualization
# let's visualize highway-mpg as potential predictor variable of price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
#We can see from this plot that price is negatively correlated to highway-mpg, since the regression 
#slope is negative. One thing to keep in mind when looking at a regression plot is to pay attention 
#to how scattered the data points are around the regression line. This will give you a good indication 
#of the variance of the data, and whether a linear model would be the best fit or not. If the data is too 
#far off from the line, this linear model might not be the best model for this data.

#Residual plot
#The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). 
#When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.

#We look at the spread of the residuals:

#If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate 
#for the data. Why is that? Randomly spread out residuals means that the variance is constant, and thus the 
#linear model is a good fit for this data.
plt.figure(figsize=None)
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
#What is this plot telling us?
#We can see from this residual plot that the residuals are not randomly spread around the x-axis, which 
#leads us to believe that maybe a non-linear model is more appropriate for this data.

#Multiple regression
#One way to look at the fit of the model is by looking at the distribution plot: We can look at the distribution 
#of the fitted values that result from the model and compare it to the distribution of the actual values.

#First let's make a prediction
Yhat = lm.predict(Z)
plt.figure(figsize=None)
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
#We can see that the fitted values are reasonably close to the actual values, since the two distributions 
#overlap a bit. However, there is definitely some room for improvement.

#Polynomial Regression and Pipelines
#Polynomial regression is a particular case of the general linear regression model or multiple linear regression 
#models. We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.

#Let's define a small function to solve it
def PlotPolly(model,independent_variable,dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable,dependent_variabble,'.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.898, 0.898, 0.898))
    plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
print("done")

#Let's get the variables
x = df['highway-mpg']
y = df['price']
print("done")

#Let's fit the polynomial using the function polyfit, then use the function poly1d to display 
#the polynomial function
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p,x,y, 'highway-mpg')

#Let's try with order = 11
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)

PlotPolly(p1,x,y, 'highway-mpg')
#The result looks better.
#We create a PolynomialFeatures object of degree 2:
pr=PolynomialFeatures(degree=2)
print(pr)
Z_pr=pr.fit_transform(Z)
#The original data is of 201 samples and 4 features
print("The original data is of :"+ str(Z.shape))
#after the transformation, there 201 samples and 15 features
print("After the transformation there are : "+ str(Z_pr.shape))

#Pipeline
#Data Pipelines simplify the steps of processing the data

#We create the pipeline, by creating a list of tuples including the name of the model or estimator 
#and its corresponding constructor.
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
print(pipe)
#We can normalize the data, perform a transform and fit the model simultaneously.
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]

#Measures for In-Sample Evaluation
#When evaluating our models, not only do we want to visualise the results, but we also want a quantitative 
#measure to determine how accurate the model is.
#Two very important measures that are often used in Statistics to determine the accuracy of a model are:

#R^2 / R-squared
#R squared, also known as the coefficient of determination, is a measure to indicate how close the data 
#is to the fitted regression line. The value of the R-squared is the percentage of variation of the response 
#variable (y) that is explained by a linear model.

#Mean Squared Error (MSE)
#The Mean Squared Error measures the average of the squares of errors, that is, the difference between actual 
#value (y) and the estimated value (ŷ)

#Model 1 : SLR
#Let's calculate the R^2
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print("R-squared with SLR = "+ str(lm.score(X, Y)))
#We can say that ~ 49.659% of the variation of the price is explained by this simple linear model "horsepower_fit".
#Let's calculate the MSE

#We can predict the output i.e., "yhat" using the predict method, where X is the input variable:
Yhat=lm.predict(X)
print(Yhat[0:4])

#mean_squared_error(Y_true, Y_predict)
print("SLR")
print("Mean squared error between true value of price and the fitted value"+ str(mean_squared_error(df['price'], Yhat)))

#Model 2 : Multiple Linear Regression
# fit the model 
lm.fit(Z, y)
# Find the R^2
print("R-squared with MLR = "+ str(lm.score(Z, y)))
#We can say that ~ 80.896 % of the variation of price is explained by this multiple linear regression "multi_fit".

#Let's calculate the MSE

#we produce a prediction
Y_predict_multifit = lm.predict(Z)

#we compare the predicted results with the actual results
print("MLR")
print("Mean squared error between true value of price and the fitted value"+ str(mean_squared_error(y, Y_predict_multifit)))
#Model 3: Polynomial Fit

#Let's calculate the R^2
r_squared = r2_score(y, p(x))
print("R-squared with Polynomial fit = "+ str(r_squared))
#We can say that ~ 67.419 % of the variation of price is explained by this polynomial fit

#We can also calculate the MSE:
print("Polynomial fit")
print("Mean squared error between true value of price and the fitted value"+ str(mean_squared_error(df['price'], p(x))))


#Prediction and Decision Making
#Let's create a new input
new_input=np.arange(1,100,1).reshape(-1,1)
#fit the model
lm.fit(X, Y)
print(lm)
#Produce a prediction
yhat=lm.predict(new_input)
yhat[0:5]
plt.plot(new_input,yhat)
plt.show()
#Now that we have visualized the different models, and generated the R-squared and MSE values for the fits, 
#how do we determine a good model fit?
#When comparing models, the model with the higher R-squared value is a better fit for the data
#When comparing models, the model with the smallest MSE value is a better fit for the data.
#### Let's take a look at the values for the different models we get.

#Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.

#R-squared: 0.49659118843391759
#MSE: 3.16 x10^7

#Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.

#R-squared: 0.80896354913783497
#MSE: 1.2 x10^7


#Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.

#R-squared: 0.6741946663906514
#MSE: 2.05 x 10^7

#Simple Linear Regression model (SLR) vs Multiple Linear Regression model (MLR)
#to be able to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared 
#and MSE to make the best conclusion about the fit of the model.

#*MSE * The MSE of SLR is 3.16x10^7 while MLR has an MSE of 1.2 x10^7. The MSE of MLR is much smaller.
#R-squared: In this case, we can also see that there is a big difference between the R-squared of the SLR 
#and the R-squared of the MLR. The R-squared for the SLR (~0.497) is very small compared to the R-squared 
#for the MLR (~0.809).
#This R-squared in combination with the MSE show that MLR seems like the better model fit in this case, 
#compared to SLR.

#Simple Linear Model (SLR) vs Polynomial Fit
#MSE: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.

#R-squared: The R-squared for the Polyfit is larger than the R-squared for the SLR, so the Polynomial Fit 
#also brought up the R-squared quite a bit.

#Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better 
#fit model than the simple linear regression for predicting Price with Highway-mpg as a predictor variable.

#Multiple Linear Regression (MLR) vs Polynomial Fit
#MSE: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
#R-squared: The R-squared for the MLR is also much larger than for the Polynomial Fit.


#Conclusion
#Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from 
#our dataset. This result makes sense, since we have 27 variables in total, and we know that more than one of 
#those variables are potential predictors of the final car price.
