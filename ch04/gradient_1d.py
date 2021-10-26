import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):       # 수치 미분을(근사로 구한 접선) 구하는 함수, x에 대한 f(x)의 변화량(즉 x일 때 기울기)
    h = 1e-4            #0.0001, h를 아주 미세한 값을 사용
    return (f(x+h) - f(x-h)) / (2*h)  # x를 중심으로 그 전후인 (x+h), (x-h)의 차분을 계산. --> 중심 차분 혹은 중앙 차분
                                      
def function_1(x):  
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):     # 수치 미분을 기울기로 하는 직선을 그리는 함수 -> 함수의 접선에 해당
    d = numerical_diff(f, x)        # x에 대한 해당함수의 변화량(기울기)
    print(d)
    y = f(x) - d*x                  # d를 기울기로 하는 접선이 y절편 구하기
    return lambda t:d*t + y         # 기울기(d), y 절편이 y 인 직선을 리턴

x = np.arange(0.0, 20, 0.1)     # (0~20) 까지 0.1 간격으로 배열 생성
y = function_1(x)               
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)       # function_1에서 x=5에 해당하는 접선
y2 = tf(x)                             
    
plt.plot(x, y)                          # original 함수
plt.plot(x, y2)                         # fuction1의 접선
plt.show()
