import tensorflow as tf
import numpy as np

# input 과 output 사이 layer를 더 쌓을 생각

# training dataset
x_data  =  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y_data = [
    [0],
    [1],
    [1],
    [0]
]

# placeholder
X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)


W1 = tf.Variable(tf.random_normal([2, 10], name='weight1'))
b1 = tf.Variable(tf.random_normal([10], name='bias1'))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 20], name='weight2'))
b2 = tf.Variable(tf.random_normal([20], name='bias2'))
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([20, 10], name='weight3'))
b3 = tf.Variable(tf.random_normal([10], name='bias3'))
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([10, 1], name='weight4'))
b4 = tf.Variable(tf.random_normal([1], name='bias4'))

# 가설 설정
logits = tf.matmul(layer3, W4) + b4
H = tf.sigmoid(logits)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
    if step % 3000 == 0:
        print(f'cost: {cost_val}')

# predcit & accuracy
predict = tf.cast(H > 0.5, dtype=tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

result = sess.run(H, feed_dict={X:x_data})
print(list(map(lambda x: int(np.round(x)), result)))
print('accuracy: {}'.format(sess.run(accuracy, feed_dict={X: x_data, Y: y_data})))