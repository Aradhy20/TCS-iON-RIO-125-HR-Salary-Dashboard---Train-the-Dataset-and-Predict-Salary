import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('Salary.csv')
data.shape
data.head
# # Check null value is present or not in dataset 
data.isnull()
data.isnull().sum()
data.info()
data.describe()
# # Prepare data
X = data.drop('Salary',axis=1)
y = data['Salary']
X.shape , y.shape
# # dataset visual
plt.scatter( data['YearsExperience'] ,data['Salary'] )
plt.xlabel(' Year of Exprience1')
plt.ylabel('Salary1')
plt.show()
# # Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) # Training and TEsting set sllit
X_train.shape , X_test.shape , y_train.shape , y_test.shape
# # Linear Regression Model
lr = LinearRegression()  # Creating objct for Linear REgression
lr.fit(X_train, y_train) # Training the model
y_test
pred = lr.predict(X_test)
pred
# ## Accuracy of linear Regression
lr.score(X_test , y_test)
# # Check Actual data , Predicted data and difference between the Actual and Predicted data
diff = y_test - pred
pd.DataFrame(np.c_[y_test , pred , diff] , columns=['Actual','Predicted','Difference'])
# # Visualize Model, that how it is performing on training data
plt.scatter(X_train , y_train , color='blue')
plt.plot(X_train ,lr.predict(X_train),color='red')
plt.title('Salary vs Experience2')
plt.xlabel('Years of Experience2')
plt.ylabel("Salary2")
plt.show()

# # Visualize Model, that how it is performing on testing data
plt.scatter(X_test , y_test,color='blue')
plt.plot(X_test ,lr.predict(X_test) ,color='red')
plt.title('Salary vs Experience3')
plt.xlabel('Years of Experience3')
plt.ylabel("Salary3")
plt.show()
R2 = r2_score(y_test,pred)
R2
# # Test on the custom data
exp =float(input("Enter year of experience"))
lr.predict([[exp]])[0]
print(f"Salary of {exp} year experience employee = {int(lr.predict([[exp]])[0])} Rupees")
# # Decision Tree
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings("ignore")
data1= pd.read_csv('Salary.csv')
data1.shape
a = data1.drop('Salary',axis=1)
b = data1['Salary']
X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.4, random_state=101)
X_train.shape , X_test.shape , y_train.shape , y_test.shape
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
pred
diff = y_test - pred
pd.DataFrame(np.c_[y_test , pred , diff] , columns=['Actual','Predicted','Difference'])
R2 = r2_score(y_test,pred)
R2
# Import required librari
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")
# Load dataset
data2 = pd.read_csv('Salary.csv')
# Split dataset into features and target variable
X = data2.drop('Salary',axis=1)
y = data2['Salary']
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)
# Create random forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Predict on the testing data
y_pred = rf.predict(X_test)
# Evaluate the model
R2 = r2_score(y_test,y_pred)
# Import necessary libraries
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv('Salary.csv')
# Load the data
X = data.drop('Salary',axis=1)
X = data.iloc[:, 0:1].values
y = data.iloc[:, 1].values
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Fit the polynomial regression model
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)
# Predict salary for test set
y_pred = poly_reg.predict(poly.fit_transform(X_test))
# Calculate R-squared score for test set
r2_test = r2_score(y_test, y_pred)
# Predict salary for custom data
custom_data = np.array([[5], [7], [10]]) # years of experience
custom_data_pred = poly_reg.predict(poly.fit_transform(custom_data))
r2_test
custom_data = np.array([[1]]) # years of experience
custom_data_pred = poly_reg.predict(poly.fit_transform(custom_data))
custom_data_pred
# Load the data
data = pd.read_csv('Salary.csv')
X = data.iloc[:, 0:1].values
y = data.iloc[:, 1].values
# Split the data into training and testing sets
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Fit the Lasso regressi model
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_poly, y_train)
# Predict salary for test set
y_pred = lasso_reg.predict(poly.fit_transform(X_test))
# Calculate R-squared score for test set
r2_test = r2_score(y_test, y_pred)
# Predict salary for custom data
custom_data = np.array([[1], [7], [10]]) # years of experience
custom_data_pred = lasso_reg.predict(poly.fit_transform(custom_data))
custom_data_pred
r2_test
# Load data
data = pd.read_csv('salary.csv')
X = data.drop('Salary',axis=1)
y = data['Salary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict the test set results and calculate accuracy
y_pred = log_reg.predict(X_test)
#accuracy 
r2_test = r2_score(y_test, y_pred)
exp =float(input("Enter year of experience"))
log_reg.predict([[exp]])[0]
print(f"Salary of {exp} year experience employee = {int(log_reg.predict([[exp]])[0])} Rupees")
from sklearn.metrics import mean_squared_error

# Load data
data = np.loadtxt('Salary.csv', delimiter=',', skiprows=1)
X = data[:, :-1]  # Input features
y = data[:, -1]  # Target variable

# Train-test split
train_size = int(0.8 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)

# Decision tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)

# Random forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)

# Polynomial regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
pr = LinearRegression()
pr.fit(X_poly_train, y_train)
pr_pred = pr.predict(X_poly_test)
pr_mse = mean_squared_error(y_test, pr_pred)

# Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)

# Plot results
models = ['LR', 'DT', 'RF', 'PR', 'LASSO']
mse_scores = [lr_mse, dt_mse, rf_mse, pr_mse, lasso_mse]
plt.bar(models, mse_scores)
plt.ylabel('MSE')
plt.title('Regression Model Performance')
plt.show()
# Load the dataset
data = pd.read_csv('Salary.csv')

# Split the data into training and testing sets
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_y_pred = lin_reg.predict(X_test)
lin_reg_r2 = r2_score(y_test, lin_reg_y_pred)

# Fit decision tree regression model
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X_train, y_train)
dt_reg_y_pred = dt_reg.predict(X_test)
dt_reg_r2 = r2_score(y_test, dt_reg_y_pred)

# Fit random forest regression model
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X_train, y_train)
rf_reg_y_pred = rf_reg.predict(X_test)
rf_reg_r2 = r2_score(y_test, rf_reg_y_pred)

# Fit polynomial regression model
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)
pol_reg_y_pred = pol_reg.predict(poly_reg.transform(X_test))
pol_reg_r2 = r2_score(y_test, pol_reg_y_pred)

# Fit lasso regression model
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
lasso_reg_y_pred = lasso_reg.predict(X_test)
lasso_reg_r2 = r2_score(y_test, lasso_reg_y_pred)

#logistic
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
r2_test = r2_score(y_test, y_pred)

#Plot the R-squared scores for each model
models = ['Linear', 'DT', 'RF', 'PR', 'Lasso','logistic']
r2_scores = [lin_reg_r2, dt_reg_r2, rf_reg_r2, pol_reg_r2, lasso_reg_r2,r2_test]
plt.bar(models, r2_scores)
plt.title('R-squared scores for different regression models')
plt.xlabel('Model')
plt.ylabel('R-squared score')
plt.show()