from matplotlib.pyplot import grid
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# 편미분을 동시에 계산 -> 모든 변수의 편미분을 벡터로 정리한 것 : 기울기
def _numerical_gradient_no_batch(f, x):     # f함수, x 넘파이 배열 -> 넘파이 배열 x의 각 원소에 대해 수치미분을 구함
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):      # 넘파이 배열 x의 사이즈만큼 수치미분 계산
        tmp_val = x[idx]            # 원래 x는 tmp_val에 잠시 보관
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h     # x에 h를 더해 (x+h)로
        fxh1 = f(x)
        
        # f(x-h) 계산                   
        x[idx] = tmp_val - h            # x에 h를 빼 (x-h)로
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)   # x를 중심으로 그 전후의 차분을 계산 -> x에 대한 f(x)의 변화량(즉 x일 때 기울기)
        x[idx] = tmp_val # 값 복원            # 원래 x를 복원
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:     # x가 1차원 리스트
        return _numerical_gradient_no_batch(f, X)   # 1차원에 대해 편미분 계산
    else:       # 1차원 리스트가 아니라 여러 개가 하나로 묶어서 들어왔다면 
        grad = np.zeros_like(X)    # 0으로 이루어진 X와 같은 형상의 배열을 만든다.
        
        for idx, x in enumerate(X): # 넘파이 배열 X의 묶음 개수만큼 수치미분 계산
            grad[idx] = _numerical_gradient_no_batch(f, x)  # 인덱스에 해당하는 x 리스트의 수치미분을 계산
        
        return grad


def function_2(x):      # 변수가 2개인 함수, ex) f(x1, x2)
    if x.ndim == 1:             # x가 1차원 배열
        return np.sum(x**2)     # x[0]**2 + x[1]**2 와 같음
    else:                       # 1차원 배열이 아니라 여러 개가 묶어서 들어왔다면
        return np.sum(x**2, axis=1)     # axis = 1, 즉 column 기준으로 x**2 한 값을 더해준다.


def tangent_line(f, x):     # 수치 미분을 기울기로 하는 직선을 그리는 함수 -> 함수의 접선에 해당
    d = numerical_gradient(f, x)    # x에 대한 해당함수의 변화량(기울기)
    print(d)
    y = f(x) - d*x                  # d를 기울기로 하는 접선의 y절편 구하기
    return lambda t: d*t + y        # 기울기(d), y 절편이 y 인 직선을 리턴
     
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)   # (-2 ~ 2.5 사이를 0.25 간격으로)
    x1 = np.arange(-2, 2.5, 0.25)   # (-2 ~ 2.5 사이를 0.25 간격으로)
    X, Y = np.meshgrid(x0, x1)      # x0, x1의 범위에 해당하는 격자그리드
    
    X = X.flatten()                 #  다차원 배열인 X, Y를 1차원 배열로 바꿈
    Y = Y.flatten()
    
    grad = numerical_gradient(function_2, np.array([X, Y]) )        # 함수2를 X, Y에 대해서 편미분 계산
    # X, Y에 대해 편미분 한 기울기 배열을 가지게 된다. --> grad = [[X의 편미분 기울기], [Y의 편미분 기울기]]
    
    print(grad)
    # print(np.shape(grad))  # (2, 324)
    plt.figure()
    # X, Y의 편미분 계산한 기울기, 즉 벡터를 2차원 평면상에 scaled된 화살표로 그려준다.
    # -grad[0], -grad[1] : 화살표의 방향
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])       # x 범위
    plt.ylim([-2, 2])       # y 범위
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()              # 격자 그리드  
    plt.legend()            # 그래프의 범례를 추가하는 것
    plt.draw()
    plt.show()