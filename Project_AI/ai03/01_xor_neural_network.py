import tensorflow as tf


#training dataset 
x_data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

y_data= [
    [0],
    [1],
    [1],
    [0]   
]


#placeholder 
X = tf.placeholder(shape = [None, 2], dtype= tf.float32)
Y = tf.placeholder(shape = [None, 1], dtype= tf.float32)


# Weight & bias 
W1 = tf.Variable(tf.random_normal([2,2], name='weigtht1'))
b1 = tf.Variable(tf.random_normal([2], name='bias1'))    

layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2,1],name='weight2'))
b2 = tf.Variable(tf.random_normal([1], name='bias2'))

#Hypothesis
logits = tf.matmul(layer1, W2) + b2
H = tf.sigmoid(logits)


#cost
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels=Y))

#train
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
                 
#session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#학습 
for step in range(30000):
    _, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})
    if step % 3000 == 0:
        print(f'cost: {cost_val}')
        
        
#predict & accuracy 
predict = tf.cast(H>0.5, dtype=tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype= tf.float32))

print(sess.run(H, feed_dict={X:x_data}))
print('accuracy: {}'.format(sess.run(accuracy, feed_dict={X:x_data, Y:y_data})))        



   # accuracy: 0.5





















 