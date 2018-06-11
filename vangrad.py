"""
try code from
https://cs224d.stanford.edu/notebooks/vanishing_grad_example.html

2018-06-11 martin
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

N = 100  # points per class
D = 2  # dimensions
K = 3  # number of classes
X = np.zeros((N * K, D))
num_train_examples = X.shape[0]
y = np.zeros(N * K, dtype='uint8')


for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(.0, 1, N)
    theta = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * .2
    X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
    y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
plt.savefig('/dev/shm/vangrad_000_spiral_scatter.png')


def relu(x):
    return np.maximum(0, x)


def pluck(dict, *args):
    return (dict[arg] for arg in args)


def train_3layer(X, y, model, step_size, reg):
    h, h2, W1, W2, W3, b1, b2, b3 = pluck(
        model, 'h', 'h2', 'W1', 'W2', 'W3', 'b1', 'b2', 'b3')
    num_examples = X.shape[0]
    for i in range(50000):
        hidden_layer = relu(np.dot(X, W1) + b1)
        hidden_layer2 = relu(np.dot(hidden_layer, W2) + b2)
        scores = np.dot(hidden_layer2, W3) + b3
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # NxK
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = .5 * reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print('iteration {}: loss {}'.format(i, loss))

        # gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # backprop

        dW3 = (hidden_layer2.T).dot(dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)

        dhidden2 = np.dot(dscores, W3.T)
        dhidden2[hidden_layer2 <= 0] = 0
        dW2 = np.dot(hidden_layer.T, dhidden2)

        # update layer 2
        # np.sum(np.abs(dW2))/np.sum(np.abs(dW2.shape))

        db2 = np.sum(dhidden2, axis=0)
        dhidden = np.dot(dhidden2, W2.T)
        dhidden[hidden_layer <= 0] = 0

        dW1 = np.dot(X.T, dhidden)

        # update layer1
        # np.sum(np.abs(dW1))/np.sum(np.abs(dW1.shape))

        db1 = np.sum(dhidden, axis=0)

        # regularization
        dW3 += reg * W3
        dW2 += reg * W2
        dW1 += reg * W1

        # update
        W1 += -step_size * dW1
        W2 += -step_size * dW2
        W3 += -step_size * dW3
        b1 += -step_size * db1
        b2 += -step_size * db2
        b3 += -step_size * db3

    # training set accuracy
    hidden_layer = relu(np.dot(X, W1) + b1)
    hidden_layer2 = relu(np.dot(hidden_layer, W2) + b2)

    scores = np.dot(hidden_layer2, W3) + b3
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: {}'.format(np.mean(predicted_class == y)))
    return W1, W2, W3, b1, b2, b3


#N = 100
#D = 2
#K = 3
h = 11
h2 = 3
num_train_examples = X.shape[0]


model = {'h': h,
         'h2': h2,
         'W1': .1 * np.random.randn(D, h),
         'W2': .1 * np.random.randn(h, h2),
         'W3': .1 * np.random.randn(h2, K),
         'b1': np.zeros((1, h)),
         'b2': np.zeros((1, h2)),
         'b3': np.zeros((1, K)), }


W1, W2, W3, b1, b2, b3 = train_3layer(X, y, model, step_size=1e-1, reg=1e-3)


h = .02
xmi, xma = X[:, 0].min() - 1, X[:, 0].max() + 1
ymi, yma = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xmi, xma, h),
                     np.arange(ymi, yma, h))
Z = np.dot(relu(np.dot(relu(np.dot(np.c_[xx.ravel(),
                                         yy.ravel()],
                                   W1)
                            + b1), W2)
                + b2), W3) + b3
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, alpha=.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.savefig('/dev/shm/vangrad_010_spiral_decision.png')
