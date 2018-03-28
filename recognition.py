import tensorflow as tf, sys

image_path = sys.argv[1]

image_data = tf.gfile.FastGFile(image_path, 'rb').read()

label_lines = [line.rstrip() for line 
				in tf.gfile.GFile("")]

with tf.gfile.FastGFile("./tf_files/retrained_labels.txt", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	predictions = sess.run(softmax_tensor, {'DecodePng/contents:0': image_data})

	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

	for node in top_k:
		human_string = label_lines[node_id]
		score = predictions[0][node_id]
		print('%s(score = %.5f)'%(human_string, score))