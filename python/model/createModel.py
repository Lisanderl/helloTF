import tensorflow as tf

config=tf.ConfigProto(log_device_placement=True)

b = tf.placeholder(tf.float32, name="input_b")
c = tf.placeholder(tf.float32, name="input_c")
target = tf.placeholder(tf.float32, name="target")

powVal = tf.Variable(1., name='powVal')
# sqrtVal = tf.Variable(0.5, name='sqrtVal')

out = (tf.math.pow(b, powVal) + tf.math.pow(c, powVal))
out = tf.identity(out, name='output')

loss = tf.reduce_mean(target - out)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss, name='train')

init = tf.global_variables_initializer()
# Creating a tf.train.Saver adds operations to the graph to save and
# restore variables from checkpoints.
saver_def = tf.train.Saver().as_saver_def()


print('Python tf version:                       ', tf.__version__)
print()
print('Operation to initialize variables:       ', init.name)
print('Tensor to feed as input data:            ', b.name)
print('Tensor to feed as input data:            ', c.name)
print('Tensor to feed as training targets:      ', target.name)
print('Tensor to fetch as prediction:           ', out.name)
print('Operation to train one step:             ', train_op.name)
print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
print('Tensor to read value of W                ', powVal.value().name)
# print('Tensor to read value of b                ', sqrtVal.value().name)

# Save model
tf.io.write_graph(tf.get_default_graph().as_graph_def(config), "", "graph.pb", False)