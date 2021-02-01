## Chaper 01 - 헬로 파이썬



#### 넘파이 (Numpy)

- numpy.array - 배열 클래스
- array.shape - 배열의 형상
- 배열.astype(np.int) - 넘파이 배열의 자료형 변환할 때 (int)
- np.random.choice(60000, 10) - 0 이상 60000 미만의 수 중에서 무작위로 10개 선택



#### matplotlib

- 그래프 그리기

```
import matplotlib.pyplot as plt

...
plt.plot (x, y)
plt.xlabel("x") - x 축 이름
plt.ylabel("y") - y 축 이름
plt.title('') - 그래프 제목
plt.legend() - 그래프 설명 (우측 상단 모서리에)
plt.imshow() - 이미지 표시
plt.show()
```

