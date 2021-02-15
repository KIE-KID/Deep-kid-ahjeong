### Ch 05 - 오차역전파법

---

신경망의 가중치 매개변수의 기울기(= 가중치 매개변수에 대한 손실함수의 기울기) 를 **수치미분** 을 사용해 구했다.

수치미분은 단순하고 구현하기 쉽지만, 계산 시간이 오래걸린다.

이번 장에서는 가중치 매개변수의 기울기를 효율적으로 계산하는 **오차역전파법(backpropagation)** 을 배운다.

- 계산그래프를 통해 역전파를 통한 '미분'을 효율적으로 계산한다.

- **국소적 미분** 을 전달하는 원리는 **연쇄법칙** 에 따른다.

  ![](../img/ch05_back.png)

  - 얇은 선 : 순전파
  - 굵은 선 : 역전파
  - 계산 그래프를 통해서 z에 대한 x의 미분을 연쇄법칙을 통해 구할 수 있다.

  

#### 1. 덧셈 노드의 역전파

z = x+y 식의 미분은 **1** 이므로,

덧셈 노드의 역전파는 → **(상류에서 전해진 미분 * 1)**

즉, 입력된 값 그대로 다음 노드로 보내진다.

```python
class AddLayer :
	def __inint__(self):
		pass
	
	def forward(self, x, y):
		out = x + y
		return out
		
	def backward(self, dout):		# 상류에서 내려온 미분을 그대로 하류로 흘려보낸다.
		dx = dout * 1
		dy = dout * 1
		return dx, dy
```



#### 2. 곱셈 노드의 역전파

z = xy 에서 

​	x에 대한 미분 : y

​	y에 대한 미분 : x

따라서, 곱셈 노드의 역전파는 다음과 같다.

![](../img/ch05_mul.png)

즉, **서로 바꾼 값** 을 곱해서 하류로 보낸다.

또한 곱셉 노드를 구현할 때는 *순전파의 입력 신호*를 변수에 저장해 둔다.

```python
class MulLayer :
	def __init__(self):
    self.x = None
    self.y = None
   
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y
    
  def backward(self, dout): # 상류에서 내려온 미분을 x와 y 를 바꿔 곱한다.
    dx = dout * self.y
    dy = dout * self.x
    
    return dx, dy
```



#### 3. 단순한 계층 구현하기 (덧셈, 곱셈)

코드 : ch05/buy_apple_orange.py

