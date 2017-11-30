import tensorflow as tf  
import numpy as np 
import pickle
from matplotlib import pyplot as plt

alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' "
char2num = {}
num2char = {}
idx = 1

for char in alphabets:
	char2num[char] = idx
	num2char[idx] = char
	idx += 1
num2char[0] = '@'
num2char[55] = '$'

def helpers(X):
	N = len(X)
	maxlen = 0
	maxlen = max([len(i) for i in X])
	O = np.zeros(shape=(maxlen,N),dtype=np.int32)
	for i in range(N):
		for j in range(maxlen):
			if j>=len(X[i]):
				break
			else:
				O[j][i]=X[i][j]
	return O


slo = pickle.load(open('encoded_slo_data.pkl','rb'))
slo_train = slo[:-1000]
slo_test = slo[-1000:]
anv = pickle.load(open('encoded_anv_data.pkl','rb'))
anv_train = anv[:-1000]
anv_test = anv[-1000:]

tf.reset_default_graph()
sess = tf.InteractiveSession()

vocab_size = 56 #54 chars, 1 end token, 1start token 
EOS = 55
input_embedding_size = 30
encoder_hidden_units = 100
decoder_hidden_units = 100

encoder_inputs = tf.placeholder(shape=(None,None),dtype=tf.int32,name="encoder_inputs")
decoder_targets = tf.placeholder(shape=(None,None),dtype=tf.int32,name="decoder_targets")
decoder_inputs = tf.placeholder(shape=(None,None),dtype=tf.int32,name="decoder_inputs")

embeddings = tf.Variable(tf.random_uniform([vocab_size,input_embedding_size],-1.0,1.0),dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs,encoder_final_state = tf.nn.dynamic_rnn(
	encoder_cell,encoder_inputs_embedded,
	dtype=tf.float32,time_major=True)

del encoder_outputs

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs,decoder_final_state = tf.nn.dynamic_rnn(
	decoder_cell,decoder_inputs_embedded,
	initial_state=encoder_final_state,
	dtype=tf.float32,time_major=True,scope="plain_decoder")

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

log = open("output.txt", "w")
print("Format - (input, target, output)\n", file = log)

batch_size = 10
epoch = 25
end_token = slo_train[0][-1]
losses = []
for ep in range(epoch):
	loss1 = 0
	# Testing after every epoch
	log = open("output.txt", "a")
	print("Testing after Epoch ", ep, "\n", file = log)

	for i in range(5):
		encoder_inputs_ = helpers(anv_test[i:i+1])
		input1 = ''
		for k in range(len(anv_test[i])):
			input1+=num2char[anv_test[i][k]]
		target1 = ''

		for k in range(len(slo_test[i])):
			target1+=num2char[slo_test[i][k]]

		encoder_final_state_ = sess.run([encoder_final_state],{encoder_inputs:encoder_inputs_})
		output = ''
		decoder_prediction_,decoder_final_state_ = sess.run([decoder_prediction,decoder_final_state],{encoder_final_state:encoder_final_state_,decoder_inputs:helpers([[55]])})
		output+=num2char[decoder_prediction_[0][0]]
		cnt=0
		while(decoder_prediction_!=55):
			cnt+=1
			if cnt>len(input1)+10:
				break
			decoder_prediction_,decoder_final_state_ = sess.run([decoder_prediction,decoder_final_state],{encoder_final_state:decoder_final_state_,decoder_inputs:helpers(decoder_prediction_)})
			output+=num2char[decoder_prediction_[0][0]]
		log = open("output.txt", "a")
		print(input1,len(input1), file = log)
		print(target1,len(target1), file = log)
		print(output,len(output), file = log)
		print("\n", file = log)

	#Training loop
	for i in range(int(len(slo_train)/batch_size)):
		batch_anv = anv_train[i*batch_size:(i+1)*batch_size]
		batch_slo = slo_train[i*batch_size:(i+1)*batch_size]
		encoder_inputs_ = helpers(batch_anv)
		decoder_targets_ = helpers([seq+[EOS] for seq in batch_slo])
		decoder_inputs_ = helpers([[EOS]+seq for seq in batch_slo])
		_,l = sess.run([train_op,loss],{
			encoder_inputs:encoder_inputs_,
			decoder_inputs:decoder_inputs_,
			decoder_targets:decoder_targets_})
		loss1+=l
		if i%100==0:
			print('epoch:',ep,'iter:',i,'loss:',loss1/batch_size)
			if i != 0:
				losses.append(loss1/batch_size)
			loss1=0

	with open('train_loss', 'wb') as fp:
		pickle.dump(np.asarray(losses), fp)


with open('train_loss', 'rb') as fp: train_loss = pickle.load(fp)


fig = plt.figure(figsize=(20,8))
plt.plot(range(1,len(train_loss)+1), train_loss, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
plt.xlabel('Iterations', fontsize=22)
plt.ylabel('Error', fontsize=22)
#plt.ylim(0,20)
plt.title('Training Error', fontsize=26)
plt.legend()
plt.show()	























