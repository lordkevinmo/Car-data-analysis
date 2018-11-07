# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 13:13:05 2018

@author: Koffi_AGBENYA

Model Evaluation and Refinement
We have built models and made predictions of vehicle prices. Now we will determine how accurate
these predictions are.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# Import clean data 
path = path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)

df = df._get_numeric_data()


#
def DistributionPlot(RedFunction,BlueFunction,RedName,BlueName,Title ):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


def PollyPlot(xtrain,xtest,y_train,y_test,lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(),xtest.values.max()])

    xmin=min([xtrain.values.min(),xtest.values.min()])

    x=np.arange(xmin,xmax,0.1)


    plt.plot(xtrain,y_train,'ro',label='Training Data')
    plt.plot(xtest,y_test,'go',label='Test Data')
    plt.plot(x,lr.predict(poly_transform.fit_transform(x.reshape(-1,1))),label='Predicted Function')
    plt.ylim([-10000,60000])
    plt.ylabel('Price')
    plt.legend()


"""
    Training and testing
"""

y_data = df['price']
x_data = df.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


#Create LR object
lre = LinearRegression()
lre.fit(x_train[['horsepower']],y_train)

#Let's Calculate the R^2 on the test data:
r2test = lre.score(x_test[['horsepower']],y_test)
print("R-squared on test data = "+ str(r2test))
#R^2 on the train data :
r2train = lre.score(x_train[['horsepower']],y_train)
print("R-squared on test data = "+ str(r2train))

#Cross validation score
Rcross=cross_val_score(lre,x_data[['horsepower']], y_data,cv=4)
print("Cross validation score"+ str(Rcross))

#Compute average and standard deviation of our estimate
print("The mean of the folds are", Rcross.mean(),"and the standard deviation is" ,Rcross.std())

#We can also use negative squared error as a score by setting the parameter 'scoring' metric 
#to 'neg_mean_squared_error'.
NegCross = -1*cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')
print("Neg squared error = "+ str(NegCross))
#can also use the function 'cross_val_predict' to predict the output. The function splits up the data into 
#the specified number of folds, using one fold to get a prediction while the rest of the folds are used as 
#test data.
yhat=cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
#Print of first 6 lines
print(yhat[0:5])


# Model evaluation and Model Selection. 
#Overfitting, Underfitting ?

#It turns out that the test data sometimes referred to as the out of sample data is 
#a much better measure of how well our model performs in the real world.
#Let's explore Model selection with MLR
#Let's create Multiple linear regression objects and train the model using 'horsepower',
# 'curb-weight', 'engine-size' and 'highway-mpg' as features.
lr=LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_train)
#Prediction using training data:
yhat_train=lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_train[0:5])
#Prediction using test data
yhat_test=lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_test[0:5])
Title='Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution '
DistributionPlot(y_train,yhat_train,"Actual Values (Train)","Predicted Values (Train)",Title)
#Plot of predicted values using the training data compared to the training data.

#So far the model seems to be doing well in learning from the training dataset. But what happens when 
#the model encounters new data from the testing dataset? 
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
#When the model generates new values from the test data, we see the distribution of the predicted 
#values is much different from the actual target values.
#Comparing Figure 1 and Figure 2; it is evident the distribution of the test data in Figure 1 
#is much better at fitting the data. This difference in Figure 2 is apparent where the ranges 
#are from 5000 to 15 000. This is where the distribution shape is exceptionally different.

#Let's see if polynomial regression also exhibits a drop in the prediction accuracy 
#when analysing the test dataset.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
print("done")
#We will perform a degree 5 polynomial transformation on the feature 'horse power'.
pr=PolynomialFeatures(degree=5)
x_train_pr=pr.fit_transform(x_train[['horsepower']])
x_test_pr=pr.fit_transform(x_test[['horsepower']])
#Now let's create a linear regression model "poly" and train it.
poly=LinearRegression()
poly.fit(x_train_pr,y_train)

yhat=poly.predict(x_test_pr )
print(yhat[0:5])
#Let's take the first five predicted values and compare it to the actual targets.
print("Predicted values:", yhat[0:4])
print("True values:",y_test[0:4].values)
PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)

#A polynomial regression model, red dots represent training data, green dots represent test data
#, and the blue line represents the model prediction.
#We see that the estimated function appears to track the data but around 200 horsepower,
#the function begins to diverge from the data points.

#R^2 of the training data:
print("R^2 of the training data:"+ str(poly.score(x_train_pr, y_train)))
#R^2 of the test data:
print("#R^2 of the training data:"+ str(poly.score(x_test_pr, y_test)))

#We see the R^2 for the training data is 0.5567 while the R^2 on the test data was -29.87. 
#The lower the R^2, the worse the model, a Negative R^2 is a sign of overfitting.

#Let's see how the R^2 changes on the test data for different order polynomials and plot the results:
Rsqu_test=[]

order=[1,2,3,4]
for n in order:
    pr=PolynomialFeatures(degree=n)
    
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr=pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr,y_train)
    
    Rsqu_test.append(lr.score(x_test_pr,y_test))

plt.plot(order,Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')

#We see the R^2 gradually increases until an order three polynomial is used. 
#Then the R^2 dramatically decreases at four.

def f(order,test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr=PolynomialFeatures(degree=order)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    poly=LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)

interact(f, order=(0,6,1),test_data=(0.05,0.95,0.05))


#Ridge regression
#Let's perform a degree two polynomial transformation on our data.
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

#Let's create a Ridge regression object, setting the regularization parameter to 0.1
RigeModel=Ridge(alpha=0.1)

#Let's fit the model
RigeModel.fit(x_train_pr,y_train)
yhat=RigeModel.predict(x_test_pr)

#Let's compare the first five predicted samples to our test set
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)


#We select the value of Alfa that minimizes the test error, for example, we can use a for loop.
Rsqu_test=[]
Rsqu_train=[]
dummy1=[]
ALFA=5000*np.array(range(0,10000))
for alfa in ALFA:
    RigeModel=Ridge(alpha=alfa) 
    RigeModel.fit(x_train_pr,y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr,y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr,y_train))

#We can plot out the value of R^2 for different Alphas
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA,Rsqu_test,label='validation data  ')
plt.plot(ALFA,Rsqu_train,'r',label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()

#The blue line represents the R^2 of the test data, and the red line represents the R^2 
#of the training data. The x-axis represents the different values of Alfa
#The red line in the figure represents the R^2 of the test data, as Alpha increases the R^2 
#decreases; therefore as Alfa increases the model performs worse on the test data. The blue 
#line represents the R^2 on the validation data, as the value for Alfa increases the R^2 decreases.


#Grid Search
#We create a dictionary of parameter values:
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000]}]
RR=Ridge()
Grid1 = GridSearchCV(RR, parameters1,cv=4)
#Fit the model
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)
#We can obtain the estimator with the best parameters and assign it to the variable BestRR as follows:
BestRR=Grid1.best_estimator_
#We now test our model on the test data
brscore = BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_test)
print(brscore)

