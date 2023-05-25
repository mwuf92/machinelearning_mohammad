# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:34:12 2023
Machine Learning

@author: Forhad
1.
Mean: Average value
Median: Mid point value
Mode: Most common value

2.Standard deviation: meaning that most of the values are within the range from 
the mean value.

3.Variance: is another number that indicates how spread out the values are.
if take the square root of the variance, you get the standard deviation.

4.Percentile: Percentiles are used in statistics to give you a number that 
describes the value that a given percent of the values are lower than.

5.Create Big Data Set

6.Plot Histogram

7.Normal Data Distribution: learn how to create an array where the values are concentrated around a given value
is known in probability as Gaussian data distribution.

8.Random data distribution & scatter plot.

9.Linear Regression, Coefficient of correlation, Predict Future Values, Bad Fit.

10.Polynomial Regression

11.Multiple Regression

12.Scale(Standardization)

13.Train/Test

14.Decision Tree

"""

'''
#1.Mean, Mode, Median
import numpy
from scipy import stats

speed = [99,86,87,88,111,86,103,94,78,77,85,86]

#To find the mean value
x = numpy.mean(speed)

#To find the median
y = numpy.median(speed)

#To find the mode
z = stats.mode(speed)

print(x)
print(y)
print(z)
#Note:The mode() method returns a ModeResult object that contains the mode number (86), and count (how many times the mode number appeared (3)).

#To findout Standard Deviation (use of std())

import numpy

speed = [32,111,138,28,59,77,97]

x = numpy.std(speed)

print(x)


#To findout Variance (use of var())

speed = [32,111,138,28,59,77,97]

y = numpy.var(speed)

print(y)


#To find Percentile (use of percentile())

import numpy

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

x = numpy.percentile(ages, 90)

print(x)

#To create Big Data Sets (use of random())

import numpy
import matplotlib.pyplot as plt

#Create Data Set(an array containing 250 random floats between 0 and 5)

x = numpy.random.uniform(0.0, 5.0, 250)

print(x)

#To plot histogram(use of hist())

x = numpy.random.uniform(0.0, 5.0, 250)

plt.hist(x, 5)
plt.show()

#Big Data Distribution

x = numpy.random.uniform(0.0, 5.0, 100000)

plt.hist(x, 100)
plt.show()

#Gaussian Data Distribution

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

#Scatter Plot (use of scatter())

import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]

y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()

#Random data distribution & scatter plot

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 1000)

y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()

#Linear Regression

import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]

y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

#Coefficient of correlation(r = -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related)

from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]

y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)

#Predict Future Values

from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]

y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

speed = myfunc(10)

print(speed)

#Bad Fit
import matplotlib.pyplot as plt
from scipy import stats

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

#10.Polynomial Regression:
import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

#R-squared (relationship between the values of the x- and y-axis is, if there are no relationship the polynomial regression can not be used to predict anything)
#r-squared value ranges from 0 to 1, where 0 means no relationship, and 1 means 100% related

import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))

#Predict Future Values

import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

speed = mymodel(17)
print(speed)

#Bad Fit
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))

myline = numpy.linspace(2, 95, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

#11.Multiple Regression

import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[3300, 1300]])

print(regr.coef_)
print(predictedCO2)

#12.Scale(use of StandardScaler()):(In summary, the code reads data from a CSV file, selects two specific columns ('Weight' and 'Volume'), applies standardization to these columns using a StandardScaler, and then prints the standardized features to the console.)

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]

scaledX = scale.fit_transform(X)

print(scaledX)

# Predict the CO2 emission from a 1.3 liter car that weighs 2300 kilograms:

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)

#13.Train/Test:
#Create a data set illustrates 100 customers in a shop, and their shopping habits.
#Create training model
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))

myline = numpy.linspace(0, 6, 100)

plt.scatter(x, y)
plt.show()

plt.scatter(train_x, train_y)
plt.show()
plt.scatter(test_x, test_y)
plt.show()

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()
print(r2)

#Bring in the Testing Set
import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(test_y, mymodel(test_x))

print(r2)

#Predict Values

print(mymodel(5))
'''
#14.Decision Tree

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("data3.csv")

print(df)

#convert non-numerical into numerical values
d = {'UK' : 0, 'USA' : 1, 'N' : 2}
df['Nationality'] = df['Nationality'].map(d)

d = {'YES' : 1, 'NO' : 0}
df['Go'] = df['Go'].map(d)

print(df)

#separate the feature columns from the target column

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

print(X)
print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)





























