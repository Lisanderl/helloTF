import tensorflow as tf


x = tf.placeholder(tf.float32, name='x')
y_ = tf.placeholder(tf.float32, name='target')


print()
print('Python tf version:                       ', tf.__version__)
print()

# Save model
tf.io.write_graph(tf.get_default_graph().as_graph_def(), "", "graph.pb", False)
