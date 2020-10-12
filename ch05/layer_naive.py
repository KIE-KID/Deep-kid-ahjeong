class MulLayer:
    def __init__(self):
        # x, y는 순전파 시의 입력 값을 유지하기 위해(역전파 시 순방향 입력신호 필요하므로.)
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out

    def backward(self, dout):

        #dout: 상류에서 넘어온 미분
        #dout에 순전파 때의 값을 '서로 바꿔' 곱한 후 하류로 흘린다.
        dx = dout * self.y #x와 y를 바꾼다.     
        dy = dout * self.x
        
        return dx, dy

class AddLayer:
    def __init__(self):     #덧셈에서는 초기화가 필요없으므로 pass('아무것도 하지 말라')
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
