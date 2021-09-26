import sys, os
# sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append('/Users/ahjeong_park/Study/Deep-kid-ahjeong')
# print(sys.path)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax, step_function, relu, identity_function, tanh, leaky_relu, elu


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("./ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # z1 = relu(a1)
    # z1 = identity_function(a1)
    # z1 = tanh(a1)
    # z1 = leaky_relu(a1)
    # z1 = elu(a1, 0.99)
    # z1 = step_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # z2 = relu(a2)
    # z2 = identity_function(a2)
    # z2 = tanh(a2)
    # z2 = leaky_relu(a2)
    # z2 = elu(a2, 0.99)
    # z2 = step_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
batch_size = 100  #배치크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):  #range(start, end, step)
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p= np.argmax(y_batch, axis = 1) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    accuracy_cnt += np.sum(p == t[i:i+batch_size])


print("Accuracy:" + str(float(accuracy_cnt) / len(x)))