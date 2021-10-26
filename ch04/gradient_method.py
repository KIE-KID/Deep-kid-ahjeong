import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient

# 경사하강법 (gradient descent)
# 기울기를 잘 이용해 함수의 최솟값(또는 가능한 작은 값)을 찾으려는 경사법
# 주의할 점 : 기울기가 가리키는 곳에 정말 함수의 최솟값이 있는지 보장할 수 없다.

# f : 최적화하려는 함수
# lr : 학습률, 너무 크거나 너무 작으면 좋지 않다.
# step_num : 경사법에 따른 반복 횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x # 초깃값 
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )        # x가 갱신되기 전 값을 저장

        grad = numerical_gradient(f, x)     # 함수의 기울기를 구함
        x -= lr * grad                         # 기울기에 학습률을 곱한 값으로 x를 갱신

    return x, np.array(x_history)


def function_2(x):      # f(x[0], x[1])
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    # x의 초깃값을 [-3.0, 4.0] 으로 설정

lr = 0.1        #Learning Rate (학습률)
step_num = 20
# function_2에 대해 경사하강법 -> 최솟값 탐색
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

print(x_history)
# print(x_history[:,0])
# print(x_history[:,1])
plt.plot( [-5, 5], [0,0], '--b')    #--:dashed line, blue
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
