import tensorflow as tf

#tf.placeholder: 그래프를 실행 시 데이터 입력 
node1 = tf.placeholder(dtype=tf.float32)
node2 = tf.placeholder(dtype=tf.float32)
node3 = node1  + node2

sess = tf.Session()

#dictionary 값을 먹이주는 것.
print(sess.run(node3, feed_dict={node1 : [10,20,30] , node2: [40,50,60]}))


#출력값: [50. 70. 90.]
