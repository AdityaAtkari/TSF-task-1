# TSF-task-1
# TSF Task 1: Prediction using Supervised ML
## #GRIPAPRIL23
### **Simple Linear Regression**
In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.

# Author: Aditya Atkari


# Importing relevant libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import statsmodels.api as sm
import seaborn as sns
sns.set()

from scipy import stats
from scipy.stats import norm, skew

from sklearn.linear_model import LinearRegression

# Hide Warnings
import warnings
warnings.filterwarnings("ignore")

# Reading data from url
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

data.head()

# Seeing data description
data.describe()

# Checking for missing data
data.isnull().sum()

# Checking correlation between input and target
data.corr()

# Divide the dataset into inputs and targets
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  

# Visualizing the data through scatterplot
plt.scatter(X,y)
plt.xlabel('Hours',fontsize=14)
plt.ylabel('Scores',fontsize=14)
plt.title('Hours v/s Scores', fontsize=20)
plt.show()

It can be observed that data follows a linear trend. Therefore it proves that there is a positive linear relationship between the input(Hours) and the target(Scores).

# visualizing the target
sns.distplot(y, fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Scores distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(y, plot=plt)
plt.show()

print("Skewness =",data['Scores'].skew())

The target(Score) is more-or-less normally distributed with **skewness=0.234**

### Using StatsModel for plotting the Best-fit-line

# Implementing Ordinary Least Squares
X1 = sm.add_constant(X)
results = sm.OLS(y,X1).fit()
results.summary()

The p-value of the input variable is very low which proves that our input is statistically significant to predict the target. Also the R-Squared and Adjusted R-Squared are more or less same so we can tell that adding intercept causes no decrease to the predictive power of our linear model. 

# Optimal values of Slope and Intercept
results.params

# Plotting the best-fit line
plt.scatter(X,y)
yhat = 9.775803*X + 2.483673
fig = plt.plot(X,yhat,lw=4, c='red', label = 'regression line')
plt.xlabel('Hours',fontsize=20)
plt.ylabel('Scores',fontsize=20)
plt.show()

Linear regression is in its basic form the same in statsmodels and in scikit-learn. However, the implementation differs which may produce different results in edge cases, and scikit learn has in general more support for larger models. So, for purpose of generality we use **Scikit Learn Library** to evaluate our model.

### Train-Test-Split

# Importing train_test_split from sklearn
from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

The train-test-split splits the whole data into train and test sets.

### Using Linear Model from Scikit Learn Library

linear_model = LinearRegression()

# Training the model
linear_model.fit(X_train, y_train)

print("Intercept = ",linear_model.intercept_)

# Plotting the regression line
best_fit_line = linear_model.coef_*X + linear_model.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, best_fit_line);
plt.show()

linear_model.score(X_train, y_train)

The model performs very good on train data providing accuracy of about 95%

**Making Predictions for test data**

# Predicting the scores for test data
y_pred = linear_model.predict(X_test) 
y_pred

# Comparing the actual vs predicted scores
df = pd.DataFrame({'Actual Score': y_test, 'Predicted Score': y_pred})  
df 

The difference between the predicted and actual scores is very less. Therefore, we can say that our model is good in predicting scores based on studying hours. 

### Predict Score for a student who studies 9.25 hours a day

# Providing input in form of a 2-D array
hours_per_day = [[9.25]]

# Predicting the score
prediction = linear_model.predict(hours_per_day)
prediction

If a student studies for **9.25 hrs/day**, he's likely to score **93.69% marks**

### Checking Train and Test Accuracies of our Linear Model

print("Train Score =",linear_model.score(X_train, y_train))
print("Test Score =",linear_model.score(X_test, y_test))

**Train Accuracy = 95.15%** 

**Test Accuracy = 94.54%**

### Evaluating the model's performance

# importing the required metrics for evaluation
from sklearn.metrics import mean_squared_error, r2_score

print('Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test, y_pred))) 
print('R2 Score:',r2_score(y_test, y_pred)) 
