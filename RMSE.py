import numpy as np

ab = [3, 76] # 임의로 정한 기울기 a, 절편 b의 값 저장

data = [[2, 81], [4, 93], [6, 91], [8, 97]] # 공부한 시간과 이에 따른 성적 실제 값
x = [i[0] for i in data] # 공부한 시간
y = [i[1] for i in data] # 성적

# y = ax + b에 a와 b 값을 대입하여 결과를 출력하는 함수
def predict(x):
    return ab[0]*x + ab[1]

# RMSE 함수
def rmse(p, a): # 예측값 p와 실제값 a를 넣어 평균제곱근을 구함
    return np.sqrt(((p-a) ** 2).mean()) # np.sqrt()는 제곱근, **은 제곱을 구하는 뜻

# RMSE 함수를 각 y값에 대입하여 최종 값을 구하는 함수
def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

# 예측 값이 들어갈 빈 리스트를 만든다.
predict_result = []

# 모든 x 값을 한 번씩 대입하여
for i in range(len(x)):
    # 그 결과 predict_result 리스트를 완성한다.
    predict_result.append(predict(x[i]))
    print("공부한 시간 = %.f, 실제 점수 = %.f, 예측 점수 = %.f" % (x[i], y[i], predict(x[i])))

# 최종 RMSE 출력
print("rmse 최종값 : " + str(rmse_val(predict_result, y)))