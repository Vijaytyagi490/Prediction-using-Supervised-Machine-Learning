#Inporting all the required libraries
import pandas as pd
 
import matplotlib.pyplot as plt  

print("Task Submitted by :vijay tyagi\n")
print("The Sparks Foundation - Data Science & Business Analytics Internship\n")
print("Task 1 - Prediction using Supervised Machine Learning")
# Reading data from remote link
url_data = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url_data)
data.head()
print(data.shape)
data.info()

data.describe()

data.isnull().sum()

#Data Visualization
# Plotting the distribution of scores
plt.rcParams["figure.figsize"] = [16,9]
data.plot(x='Hours', y='Scores', style='*',color='blue',markersize=10)  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()

#Correlation
data.corr()

#Data Preparartion
data.head()
X = data.iloc[:, :1].values

Y = data.iloc[:, 1:].values

print(X,Y)

from sklearn.model_selection import train_test_split
X_train_set, X_test_set, Y_train_set, Y_test_set = train_test_split(X, Y,test_size=0.2, random_state=0)

#training algorithm
from sklearn.linear_model import LinearRegression  

linear_model = LinearRegression()  
linear_model.fit(X_train_set, Y_train_set)

#Visualizing the model

model_line = linear_model.coef_*X + linear_model.intercept_

plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(X_train_set, Y_train_set ,color ='red')
plt.plot(X, model_line, color='green');
plt.xlabel('Hours Studied')
plt.ylabel('Percantage Score')
plt.grid()

#Predictions

print(X_test_set)
y_prediction = linear_model.predict(X_test_set)
plt.show()

print(Y_test_set)
print(y_prediction)

computer = pd.DataFrame({ 'Actual':[Y_test_set],'Predicted':[y_prediction] })
print(computer)

hours = 9.25
prediction = linear_model.predict([[hours]])
print("The predicted score if a person studies for",hours,"hours is", prediction[0])

#Model Evaluation
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_set, y_prediction))