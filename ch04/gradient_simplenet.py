import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):   #예측 수행
        return np.dot(x, self.W)
    
    def loss(self, x, t):   #손실함수, x:입력데이터, t: 정답레이블
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
 
net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

np.argmax(p)  #최댓값의 인덱스

t = np.array([0, 0, 1)]) #정답 레이블
net.loss(x, t)