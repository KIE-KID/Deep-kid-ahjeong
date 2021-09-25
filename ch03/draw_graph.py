import sys, os
sys.path.append('/Users/ahjeong_park/Study/Deep-kid-ahjeong')
import numpy as np
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax, step_function, relu, identity_function, tanh, leaky_relu, elu
from matplotlib import pyplot as plt

# 함수 그래프
# x_values = np.arange(-10, 10, 0.1)
# y_values = elu(x_values, 0.5)
# plt.plot(x_values, y_values)	# line 그래프를 그립니다
# plt.xlabel('x')
# plt.ylabel('elu(x)')
# plt.title('elu(alpha = 0.5)')
# plt.show()	# 그래프를 화면에 보여줍니다

# 성능 막대그래프
x = np.arange(7)
function = ['softmax', 'sigmoid', 'relu', 'identity', 'tanh', 'leaky_relu', 'elu' ]
values = [0.9352]*len(function)
plt.bar(x, values, color = ['#F3FEB0', '#FEA443', '#705E78', '#A5AAA3', '#812F33','#3EB595'])
plt.xticks(x, function)
plt.ylim(0, 1)
plt.title('Compare Acitve Function')
plt.show()