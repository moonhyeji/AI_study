import tensorflow as tf


#1. training data set(학습 데이터셋)
#x = [1,2,3,4,5]
x = tf.placeholder(tf.float32)


#label (정답)
#y = [1,2,3,4,5]
y = tf.placeholder(tf.float32)

#2. Weight & bias를 정의 
W = tf.Variable(tf.random_normal([1]), name ='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#3.Hypothesis 가설 설정 
H = W *x + b 

#4. loss function 
#가설에서 정답을 빼서 제곱한 값들을 모두 더해서 평균을 낸 그래프 
loss = tf.reduce_mean(tf.square(H - y))


 
#5. gradient decent algorithm (경사하강법)
optimizer = tf.train.GradientDescentOptimizer(0.01)    #tensorflow  1.15.0 인데 왜 빨간줄 뜨는거야...  
train = optimizer.minimize(loss)   #loss값의 최소값을 구하겠다.

# 2-5 까지가 그래프 만드는 것. 


#6. 실행 
sess = tf.Session()
#초기화
sess.run(tf.global_variables_initializer())


#7.학습 
epochs = 3000

for step in range(epochs):
    tmp, loss_val, W_val, b_val = sess.run([train, loss,W, b] , feed_dict = {x: [1,2,3,4,5], y :[1,2,3,4,5]})
    if step % 300 == 0:
        print(f'Weight: {W_val} \t bias: {b_val} \t loss: {loss_val}')
    
    
    # -------가설설정함. 
    
# 출력결과 보면 Weight 가 1 에 가까워지는 것을 볼 수 있다.    
    
#8.prediction 예측 
print(sess.run(H, feed_dict={x:[10,20,30,40,50]}))


#내가 입력과 출력, 답을 미리 알려주고, 이 답을 가장 잘 서명할 수 있는 그래프를 그려주고 ㅣㅅㅍ다.
#그러면 다른 어떤 값을 넣어도 그 값에 맞는 출력을 컴퓨터가 내길 원한다.

# 그럼 값을 넣어줄 place holder를 걸어주고, 실행할 때 feed_dict 의 값으로 학습해! 라고 하는 것, 
#값이 지금은 별로 없지만 데이터가 많으면 많을수록 학습이 제대로 됨.  

#3. y =Wx + b 라는 1차함수. weight의 bias를 가지고 1차함수 만들어야 하는데, 처음에는 랜덤으로만들고,
     # -----------w 와 b 의 값을 찾아가는 과정 이 학습이다.! 그러면 나중에 우리가 x를 넣어줬을 때  y가 나올테니까.

#4. 나는 답을 알고 잇는상태이고, 컴퓨터에서 출력되어 나오는 값과 내가 알고 있는 답의 차이가 있을 건데, 이 차이를 제곱해서 더해서 평균을 냄. - 
                    #4. 그러면 실제와 출력값의 데이터의 차이가 나옴. (제곱한 값을 더해서 평균내면 차이가 나옴)
                                  #4. 1차 함수 제곱했으니까 2차함수가 나옴. 
                                  
#5.미분을 이용해서 2차함수의 경사를 점점 좁혀나감. 2차함수 기울기를 0으로 함. 
   # - 이 그래프의 기울기를 얼마만큼 내려가게 할 것인가를 정함. 

#  학습을 많이 할수록 내가 원하는 기울기가 나온다. -> 그게 w 와 b 를 찾는 행위.







