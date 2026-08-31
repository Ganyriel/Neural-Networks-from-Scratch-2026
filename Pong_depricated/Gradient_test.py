import numpy as cp
import matplotlib.pyplot as plt


def f(x):
    return 1 / (1 + cp.exp(-x))


def df(x):
    return f(x)*(1 - f(x))



def gradient_descent(starting_point, learning_rate, iterations):
    x = starting_point
    
    for i in range(iterations):
        x = x - learning_rate * df(x)  # update step
        # print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")
        learning_rate= learning_rate/1.0001
    return x

starting_point = 2
learning_rate = 0.1
iterations = 1_000_000

minimum = gradient_descent(starting_point, learning_rate, iterations)
print(f"\nLocal minimum occurs at x = {minimum:.4f}, f(x) = {f(minimum):.4f}, df(x) = ", df(minimum))


x_vals = cp.linspace(-10, 10, 1000)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label="f(x)")
plt.scatter(minimum, f(minimum), color='red', label="Local Minimum")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent Visualization")
plt.legend()
plt.show()