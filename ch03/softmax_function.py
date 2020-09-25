import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # 분모가 지수함수라 오버플로우 발생 방지를 위해서.
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
