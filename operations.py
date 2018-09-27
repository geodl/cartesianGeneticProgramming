import numpy as np

def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

def negative(x):
    return -x

def subtract(x,y):
    return x - y

def one():
    return 1

def sigmoid(*inputs):
    # print(inputs)
    z = sum(inputs)
    return 1/(1 + np.exp(-z))