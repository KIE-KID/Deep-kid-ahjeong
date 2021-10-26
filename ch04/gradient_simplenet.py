import sys, os
sys.path.append(os.pardir)
sys.path.append('/Users/ahjeong_park/Study/Deep-kid-ahjeong/')
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

# 간단한 신경망 예제
# 신경망 학습에서도 기울기를 구해야 한다.
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)       # 형상이 2*3 인 가중치 매개변수 랜덤으로 생성

    def predict(self, x):   #예측 수행        
        return np.dot(x, self.W)
    
    def loss(self, x, t):   #손실함수, x:입력데이터, t: 정답레이블
        z = self.predict(x)     
        y = softmax(z)
        loss = cross_entropy_error(y, t)        # 정답과 예측 사이의 error 

        return loss
 
x = np.array([0.6, 0.9])        # 입력데이터
t = np.array([0, 0, 1])         # 정답레이블

net = simpleNet()

f = lambda w: net.loss(x, t)    # loss 함수
dW = numerical_gradient(f, net.W)   #  가중치 매개변수에 대한 손실 함수의 기울기

print(dW)