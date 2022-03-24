import numpy as np

# x값과 y값
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x와 y의 평균값
mx = np.mean(x) # mean: 모든 원소의 평균을 구하는 넘파이 함수
my = np.mean(y)
print("x의 평균값", mx)
print("y의 평균값", my)

# 기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x]) # x의 평균값과 x의 각 원소들의 차를 제곱하라

# 기울기 공식의 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)): # d에 x의 각 원소와 평균의 차, y의 각 원소와 평균의 차를 곱해서 차례로 더하는 최소 제곱법 구현
        d += (x[i]-mx) * (y[i] - my)
    return d

dividend = top(x, mx, y, my)

print("분모:", divisor)
print("분자:", dividend)

# 기울기와 y절편 구하기
a = dividend / divisor
b = my - (mx*a)

# 출력으로 확인
print("기울기 a =", a)
print("y절편 b =", b)
