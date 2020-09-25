import sys, os
sys.path.append(os.pardir)      # 부모 디렉터리의 파일을 가져올 수 있도록 설정.
from dataset.mnist import load_mnist

# 처음 한번은 몇분정도 걸립니다.
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten = True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape)  
print(t_train.shape)
print(x_test.shape)
print(t_train.shape)
