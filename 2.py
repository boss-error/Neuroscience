import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def ForwardPass(i1, i2, w1, w2, w3, w4, b1, w5, w6, w7, w8, b2):
    h1 = sigmoid(i1 * w1 + i2 * w2 + b1)
    h2 = sigmoid(i1 * w3 + i2 * w4 + b1)
    o1 = sigmoid(h1 * w5 + h2 * w6 + b2)
    o2 = sigmoid(h1 * w7 + h2 * w8 + b2)
    return h1, h2, o1, o2
def Backpropagation(i1, i2, h1, h2, o1, o2, w5, w6, w7, w8, target_o1, target_o2, learning_rate):
    delta_o1 = (o1 - target_o1) * sigmoid_derivative(o1)
    delta_o2 = (o2 - target_o2) * sigmoid_derivative(o2)
    
    dw5 = delta_o1 * h1
    dw6 = delta_o1 * h2
    dw7 = delta_o2 * h1
    dw8 = delta_o2 * h2
    
    delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * sigmoid_derivative(h1)
    delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * sigmoid_derivative(h2)
    
    dw1 = i1 * delta_h1
    dw2 = i2 * delta_h1
    dw3 = i1 * delta_h2
    dw4 = i2 * delta_h2
    
    db1 = delta_h1 + delta_h2
    db2 = delta_o1 + delta_o2
    
    return dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, db1, db2

i1, i2 = 0.05, 0.1
w1, w2, w3, w4 = 0.15, 0.2, 0.25, 0.3
b1 = 0.35
w5, w6, w7, w8 = 0.4, 0.45, 0.5, 0.55
b2 = 0.6
target_o1, target_o2 = 0.01, 0.99
learning_rate = 0.5
h1, h2, o1, o2 = ForwardPass(i1, i2, w1, w2, w3, w4, b1, w5, w6, w7, w8, b2)

dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, db1, db2 = Backpropagation(
    i1, i2, h1, h2, o1, o2, w5, w6, w7, w8, target_o1, target_o2, learning_rate
)
w1 = w1 - learning_rate * dw1
w2 = w2 - learning_rate * dw2
w3 = w3 - learning_rate * dw3
w4 = w4 - learning_rate * dw4
w5 = w5 - learning_rate * dw5
w6 = w6 - learning_rate * dw6
w7 = w7 - learning_rate * dw7
w8 = w8 - learning_rate * dw8
b1 = b1 - learning_rate * db1
b2 = b2 - learning_rate * db2
def calculate_error(target_o1, target_o2, o1, o2):
    return 0.5 * ((target_o1 - o1) ** 2 + (target_o2 - o2) ** 2)

error = calculate_error(target_o1, target_o2, o1, o2)
print(f"Error: {error:.6f}")

print(f"Updated w1: {w1:.4f}, w2: {w2:.4f}, w3: {w3:.4f}, w4: {w4:.4f}")
print(f"Updated w5: {w5:.4f}, w6: {w6:.4f}, w7: {w7:.4f}, w8: {w8:.4f}")
print(f"Updated b1: {b1:.4f}, b2: {b2:.4f}")
