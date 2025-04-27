import numpy as np

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

text = "i love artificial intelligence".strip()
words = text.split()


unique_words = sorted(set(words))
vocab_size = len(unique_words)
word_to_idx = {word: i for i, word in enumerate(unique_words)}
idx_to_word = {i: word for i, word in enumerate(unique_words)}

def one_hot_encode(word, vocab_size):
    vec = np.zeros((vocab_size, 1))
    vec[word_to_idx[word]] = 1
    return vec

inputs = [one_hot_encode(word, vocab_size) for word in words[:3]]
targets = [one_hot_encode(word, vocab_size) for word in words[1:]]

hidden_size = 3
Wx = np.random.randn(hidden_size, vocab_size) * 0.1
Wh = np.random.randn(hidden_size, hidden_size) * 0.1
Wy = np.random.randn(vocab_size, hidden_size) * 0.1
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def forward(inputs, h_prev):
    hs = [h_prev]
    as_ = []
    ys = []
    for x in inputs:
        a_t = np.dot(Wx, x) + np.dot(Wh, h_prev) + bh
        h_t = tanh(a_t)
        y_t = softmax(np.dot(Wy, h_t) + by)
        as_.append(a_t)
        hs.append(h_t)
        ys.append(y_t)
        h_prev = h_t
    return ys, hs, as_, h_prev

def backward(inputs, targets, hs, as_, ys, learning_rate=0.01):
    global Wx, Wh, Wy, bh, by
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    dh_next = np.zeros((hidden_size, 1))
    loss = 0
    for t in reversed(range(len(targets))):
        y_pred = ys[t]
        target = targets[t]
        loss += -np.sum(target * np.log(y_pred + 1e-8))
        dy = y_pred - target
        dWy += np.dot(dy, hs[t+1].T)
        dby += dy
        dh = np.dot(Wy.T, dy) + dh_next
        da = dh * (1 - hs[t+1]**2)
        dWx += np.dot(da, inputs[t].T)
        dWh += np.dot(da, hs[t].T)
        dbh += da
        dh_next = np.dot(Wh.T, da)
    Wx -= learning_rate * dWx
    Wh -= learning_rate * dWh
    Wy -= learning_rate * dWy
    bh -= learning_rate * dbh
    by -= learning_rate * dby
    return loss

h_prev = np.zeros((hidden_size, 1))
epochs = 1000
for epoch in range(epochs):
    ys, hs, as_, h_prev = forward(inputs, h_prev)
    loss = backward(inputs, targets, hs, as_, ys)

h_prev = np.zeros((hidden_size, 1))
test_input = [one_hot_encode(word, vocab_size) for word in words[:3]]
ys, hs, as_, h_prev = forward(test_input, h_prev)
predicted_idx = np.argmax(ys[-1])
predicted_word = idx_to_word[predicted_idx]

print(f"Input text: {' '.join(words)}")
print(f"Predicted 4th word: {predicted_word}")
print(f"Full sequence: {' '.join(words[:3])} {predicted_word}")
