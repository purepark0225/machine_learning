import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# x, y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

# 학습률 값
learning_rate = 0.1

# 기울기 a 와 절편 b값을 임의로 정함.
# 기울기가 너무 커지거나 작아지면 실행 시간이 불필요하게 늘어나므로 기울기는 0~10 사이에서, y 절편은 0~100 사이에서 임의의 값 설정
a = tf.Variable(tf.random.uniform([1], 0, 10, dtype = tf.float64, seed = 0)) # 0에서 10 사이에서 임의의 수 1개를 만들어라
b = tf.Variable(tf.random.uniform([1], 0, 100, dtype = tf.float64, seed = 0)) # dtype(데이터 형식)은 실수형,
# 실행 시 같은 값이 나올 수 있게 seed 값을 설정
# Variable() 함수는 변수의 값을 정할 때 사용
# random_uniform()은 임의의 수를 생성해 주는 함수로, 몇 개의 값을 뽑아낼 지와 최솟값 및 최대값을 적어준다.

# y에 대한 일차 방정식 ax+b의 식을 세운다.
y = a * x_data + b

# 텐서플로 RMSE 함수
rmse = tf.sqrt(tf.reduce_mean(tf.square( y - y_data )))

# tf.sqrt(x): x의 제곱근을 계산
# tf.reduce_mean(x): x의 평균을 계산
# tf.square(x): x의 제곱을 계산

# RMSE 값을 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
# tensorflow의 GradientDescentOptimizer() 함수를 통해 경사 하강법의 결과를 gradient_decent에 할당

# 텐서플로를 이용한 학습
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    # 2001번 실행(0번째를 포함하므로)
    for step in range(2001):
        sess.run(gradient_decent)
        # 100번마다 결과 출력
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y 절편 b = %.4f" % (step, sess.run(rmse), sess.run(a), sess.run(b)))

# 텐서플로는 session 함수를 이용해 구동에 필요한 리소스를 컴퓨터에 할당하고 이를 실행시킬 준비를 합니다.
# Session을 통해 구현될 함수를 텐서플로에서는 '그래프' 라고 부르며, Session이 할당되면 session.run('그래프명')의 형식으로 해당 함수를 구동
# global_variables_initializer()는 변수를 초기화하는 함수.