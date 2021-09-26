import sys, os
sys.path.append('/Users/ahjeong_park/Study/Deep-kid-ahjeong')
import numpy as np
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax, step_function, relu, identity_function, tanh, leaky_relu, elu
from matplotlib import pyplot as plt

# 함수 그래프
x_values = np.arange(-10, 10, 0.1)
y_values = elu(x_values, 0.99)
plt.plot(x_values, y_values)	# line 그래프를 그립니다
plt.xlabel('x')
plt.ylabel('elu(0.99)(x)')
plt.title('elu(alpha = 0.99)')
plt.show()	# 그래프를 화면에 보여줍니다

# # 성능 막대그래프
# x = np.arange(7)
# function = ['identity', 'relu', 'leaky_relu', 'elu(0.5)', 'elu(0.99)', 'tanh', 'sigmoid' ]
# values = [0.7889, 0.8415, 0.8422, 0.8528, 0.8549, 0.8812, 0.9352]
# plt.bar(x, values, color = ['#F3FEB0', '#FEA443', '#705E78', '#A5AAA3', '#812F33','#3EB595', '#00DDDE'])
# plt.xticks(x, function)
# plt.ylim(0, 1)
# plt.title('Compare Acitve Function')
# plt.show()