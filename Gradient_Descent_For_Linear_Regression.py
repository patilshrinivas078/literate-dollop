#Gradient Descent for Linear Regression

import numpy as np


def Gradient_Descent(x,y):
    w=b=0
    iterations=1000
    m=len(x)
    learning_rate=0.1

    for i in range(iterations):
        f = w * x + b                                       #Linear Regression Model
        cost = (1/(2*m)) * sum([val**2 for val in (y-f)])  #Cost Function
        dw = (1/m) * sum(x*(f-y)) 
        db = (1/m) * sum(f-y)
        w = w - learning_rate * dw                    #Gradient Descent
        b = b - learning_rate * db
        print("w=",w,"  b=",b,"  cost=",cost)


x=np.array([1,2,3,4,5])
y=np.array([8,10,12,14,17])
Gradient_Descent(x,y)
