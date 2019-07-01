import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
data = np.loadtxt('dataset/university_admission.txt', delimiter=",") 
x = data[:, 0:2] 
y = data[:, 2] 
print(data[:5])

#scatter plot
fig, ax = plt.subplots()
positives = np.where(y == 1)
negatives = np.where(y == 0) 
ax.scatter(x[positives, 0], x[positives, 1], marker="+", c='green')
ax.scatter(x[negatives, 0], x[negatives, 1], marker="x", c='red', linewidth=1)
plt.title("University Admission", fontsize=16) 
plt.xlabel("exam 1 score", fontsize=14)
plt.ylabel("exam 2 score", fontsize=14) 
plt.legend(["admitted", "not admitted"]) 
plt.show()

def sigmoid(z):
       return 1/ (1 + np.exp(-z))
# testing the sigmoid function

def costFunction(theta, X, y):
    m=len(y)
    predictions = sigmoid(np.dot(X,theta))
    error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))
    cost = 1/m * sum(error)
    grad = 1/m * np.dot(X.transpose(),(predictions - y))
    return cost , grad

m , n = x.shape[0], x.shape[1]
X= np.append(np.ones((m,1)),x,axis=1)
y=y.reshape(m,1)

initial_theta = np.zeros((n+1,1))
cost,grad= costFunction(initial_theta,X,y)

print("Cost of initial theta is",cost)
print("Gradient at initial theta (zeros):",grad)

#optization
def costfn(theta, X, y):    
    predictions = sigmoid(X @ theta)   
    predictions[predictions == 1] = 0.999 # log(1)=0 causes division error during optimizat 
    error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions);   
    return sum(error) / len(y);
def cost_gradient(theta, X, y):   
    predictions = sigmoid(X @ theta);    
    return X.transpose() @ (predictions - y) / len(y)


y=y.reshape(m)
theta = opt.fmin_cg(costfn, initial_theta, cost_gradient, (X, y))

plt.scatter(x[positives, 0], x[positives, 1], marker="+", c='green')
plt.scatter(x[negatives, 0], x[negatives, 1], marker="x", c='red', linewidth=1)
x_value= np.array([np.min(X[:,1]),np.max(X[:,1])])
y_value=-(theta[0] +theta[1]*x_value)/theta[2]
plt.plot(x_value,y_value, "r")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)
