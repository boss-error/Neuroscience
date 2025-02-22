import numpy as np
i1, i2 = 0.05, 0.10
w1, w2, w3, w4 = np.random.uniform(-0.5, 0.5, 4)
w5, w6, w7, w8 = np.random.uniform(-0.5, 0.5, 4)
b1, b2 = 0.5, 0.7
print("weightlayer1:",[w1,w2,w3,w4])
print("weightlayer2:",[w5,w6,w7,w8])
def tanh(x):
    return np.tanh(x)
def net(input1, input2, weight1, weight2, bias):
    return input1 * weight1 + input2 * weight2 + bias
net1 = net(i1, i2, w1, w2, b1)
net2 = net(i1, i2, w3, w4, b1)
out1 = tanh(net1)
out2 = tanh(net2)
net3 = net(out1, out2, w5, w6, b2)
net4 = net(out1, out2, w7, w8, b2)
out3 = tanh(net3)
out4 = tanh(net4)
print("o1:", out3)
print("o2:", out4)
actual_out3, actual_out4 = 0.01, 0.99
error = 0.5 * ((out3 - actual_out3) ** 2 + (out4 - actual_out4) ** 2)
print("Error:", error)
