# Lab 2

## Question 1

Regardless whether we know or not the shape of the distribution of a random variable, an interval centered around the mean whose total length is 8 standard deviations is guaranteed to include at least a certain percantage of data. This guaranteed minimal value as a percentage is

```markdown
k = 8
(1-1/(k*k))*100
```
### 98.4375


## Question 3
In the 'mtcars' dataset the zscore of an 18.1mpg car is 

```markdown
data = pd.read_csv('mtcars.csv')
import pandas as pd
import numpy as np
import scipy.stats as stats
mpg = data['mpg']
stats.zscore(mpg)
```

### z-score of an 18.1mpg car is -0.33557233




## Question 4
In the 'mtcars' dataset determine the percentile of a car that weighs 3520bs is (round up to the nearest percentage point)

```markdown
weight = data['wt']
stats.percentileofscore(weight, 3.520)
```

### Percentile of car that weights 3520lbs is 68.75; rounds up to 69




## Question 6
For the 'mtcars' data set use a linear model to predict the mileage of a car whose weight is 2800lbs. The answer with only the first two decimal places and no rounding is:

```markdown
from sklearn import linear_model
from sklearn.preprocessing import QuantileTransformer
qtn = QuantileTransformer(n_quantiles=100)
x = data[['wt']]
y = data[['mpg']]
lm = linear_model.LinearRegression()
model = lm.fit(x,y)
wtp = qtn.fit_transform(x)

lm.predict([[2.8]])
```

### Linear model predicts milegage of 22.32 for a car whose weight is 2800lbs




## Question 7
In this problem you will use the gradient descent algorithm as presented in the 'Linear-regression-demo' notebook. For the 'mtcars' data set if the input variable is the weight of the car and the output variable is the mileage, then (slightly) modify the gradient descent algorithm to compute the minimum sum of squared residuals. If, for running the gradient descent algorithm, you consider the learning_rate = 0.01, the number of iterations = 10000 and the initial slope and intercept equal to 0, then the optimal value of the sum of the squared residuals is

```markdown
x = cars[['wt']]
y = cars[['mpg']]

learning_rate = 0.01
initial_b = 0
initial_m = 0
num_iterations = 10000
data = np.concatenate((x.values,y.values),axis=1)

# the average of the squared residuals.
def compute_cost(b, m, data):
    total_cost = 0
    
    # number of datapoints in training data
    N = float(len(data))
    
    # Compute sum of squared errors
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        total_cost += (y - (m * x + b)) ** 2
        
    # Return average of squared error
    return total_cost/(2*N)
    
  def step_gradient(b_current, m_current, data, alpha):
    """takes one step down towards the minima
    
    Args:
        b_current (float): current value of b
        m_current (float): current value of m
        data (np.array): array containing the training data (x,y)
        alpha (float): learning rate / step size
    
    Returns:
        tuple: (b,m) new values of b,m
    """
    
    m_gradient = 0
    b_gradient = 0
    N = float(len(data))

    # Calculate Gradient
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        m_gradient += - (2/N) * x * (y - (m_current * x + b_current))
        b_gradient += - (2/N) * (y - (m_current * x + b_current))
    
    # Update current m and b
    m_updated = m_current - alpha * m_gradient
    b_updated = b_current - alpha * b_gradient

    #Return updated parameters
    return b_updated, m_updated

def gradient_descent(data, starting_b, starting_m, learning_rate, num_iterations):
    """runs gradient descent
    
    Args:
        data (np.array): training data, containing x,y
        starting_b (float): initial value of b (random)
        starting_m (float): initial value of m (random)
        learning_rate (float): hyperparameter to adjust the step size during descent
        num_iterations (int): hyperparameter, decides the number of iterations for which gradient descent would run
    
    Returns:
        list : the first and second item are b, m respectively at which the best fit curve is obtained, the third and fourth items are two lists, which store the value of b,m as gradient descent proceeded.
    """

    # initial values
    b = starting_b
    m = starting_m
    
    # to store the cost after each iteration
    cost_graph = []
    
    # to store the value of b -> bias unit, m-> slope of line after each iteration (pred = m*x + b)
    b_progress = []
    m_progress = []
    
    # For every iteration, optimize b, m and compute its cost
    for i in range(num_iterations):
        cost_graph.append(compute_cost(b, m, data))
        b, m = step_gradient(b, m, data, learning_rate)
        b_progress.append(b)
        m_progress.append(m)
        
    return [b, m, cost_graph,b_progress,m_progress]
  
  b, m, cost_graph,b_progress,m_progress = gradient_descent(data, initial_b, initial_m, learning_rate, num_iterations)

#Print optimized parameters
print ('Optimized b:', b)
print ('Optimized m:', m)

#Print error with optimized parameters
print ('Minimized cost:', compute_cost(b, m, data))

```
### Optimized b: 37.285117303091454
### Optimized m: -5.344469026915932
### Minimized cost: 4.348780274117971
