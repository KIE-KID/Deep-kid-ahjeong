import sys, os
sys.path.append(os.curdir)
from common.functions import *
from common.gradient import numerical_gradient

# 2층 신경망 클래스
class TwoLayerNet:
    # 초기화(입력층 뉴런 수, 은닉층 뉴런 수, 출력층 뉴런 수)
    def __init__(self, input_size, hidden_size, output_size,
                weight_init_std=0.01):
                #가중치 초기화
                self.params = {}    # 신경망의 매개변수 보관 딕셔너리
                # W1: 첫번째 층 가중치
                self.params['W1'] = weight_init_std * \
                    np.random.randn(input_size, hidden_size)   
                # b1: 첫번째 층 편향
                self.params['b1'] = np.zeros(hidden_size)
                # W2: 두번째 층 가중치
                self.params['W2'] = weight_init_std * \
                    np.random.randn(hidden_size, output_size)
                # b2: 두번째 층 편향
                self.params['b2'] = np.zeros(output_size)

    # 입력 데이터 x 에 대한 예측 수행
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1 
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    # x:입력 데이터, t:정답 레이블 의 손실 함수 --> 크로스 엔트로피 사용
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 정확도
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        # 정답레이블, 예측이 같은 것의 비율
        accuracy = np.sum(y == t) / float(x.shape[0])   
        return accuracy
    
    # x: 입력데이터, t: 정답레이블의 가중치 매개변수의 기울기
    # 이 함수는 '수치 미분 방식' 으로 매개변수의 기울기를 계산함.
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        # 각 매개변수의 손실 함수에 대한 기울기 계산
        grads = {}      # 기울기 보관하는 딕셔너리 변수
        # 1번째 층의 가중치의 기울기
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 1번째 층의 편향 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 2번째 층의 가중치의 기울기
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 2번째 층의 편향 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads