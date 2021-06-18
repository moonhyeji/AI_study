import tensorflow as tf

# tf.float32 == np.float32 -> tf 내부적으로 numpy 사용한다.
node1 = tf.constant(10, dtype=tf.float32)
node2 = tf.constant(20, dtype=tf.float32)


#tensor = 뉴런 = 인공신경막 . 만들어서 자극을 줘서 전달주는 것이기 때문에 느려보임. 

node3 = node1 + node2
sess = tf.Session()


print(sess.run(node3))