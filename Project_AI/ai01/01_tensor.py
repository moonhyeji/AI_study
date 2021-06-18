import tensorflow as tf

#print(tf.__version__)


#상수 노드
node = tf.constant(100)

#session
sess = tf.Session()


#실행 
print(sess.run(node))