import pandas as pd
import numpy
import csv

# importing os functionality
import datetime
from os.path import dirname
from os.path import join
from os.path import realpath

# importing utilities for TensorFlow Keras boilerplate code
import params_flow as pf
# shortening tensorflow for certain calls
import tensorflow as tf
from bert import BertModelLayer
from bert import fetch_google_bert_model
from bert import load_stock_weights, params_from_pretrained_ckpt
from bert.tokenization.bert_tokenization import FullTokenizer
from params_flow import Concat
from params_flow.optimizers import RAdam
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
# importing TensorFlow eras API
from tensorflow.keras.models import Sequential

samples = './data/SMSSpam'
model_name = "uncased_L-12_H-768_A-12"
model_folder = "bert_models"

adapted_model = 'adapted' + model_name + '.h5'
output_folder = "."

batch_size = 16
max_seq_len = 100
adapter_size = 4
total_epoch_count = 2
buffer_size = 47
expected_number_spam = 747
print(f"\n fetching google pre-trained BERT model {model_name}")
model_dir = fetch_google_bert_model(model_name, model_folder)

bert_ckpt_file = join(model_dir, 'bert_model.ckpt')


def parse_raw_to_csv(raw_file='SMSSpam'):
	print(f"\n parsing {raw_file} into SMSSpam.csv ...")
	infile = open(raw_file, 'r')
	outfile = open('./data/SMSSpam.csv', 'w', newline='')
	columns = ['label', 'feature']
	spamwriter = csv.DictWriter(outfile, fieldnames=columns)
	spamwriter.writeheader()
	row = dict().fromkeys(columns)
	for line in infile:
		words = line.split()
		row['label'] = words[0]
		row['feature'] = ' '.join(words[1:])
		spamwriter.writerow(row)
	outfile.close()
	infile.close()
	return raw_file + '.csv'


def split_tfdataset_train_test(filename):
	file_name = parse_raw_to_csv(raw_file=filename)
	df = pd.read_csv(file_name)
	df_train, df_test = train_test_split(df, train_size=0.2, random_state=42)
	train_file_name = './data/train_SMSSpam'
	test_file_name = './data/test_SMSSpam'
	numpy.savetxt(train_file_name, df_train.values, fmt='%s %s')
	numpy.savetxt(test_file_name, df_test.values, fmt='%s %s')

	return './data/train_SMSSpam', './data/test_SMSSpam'


train_samples_file, test_samples_file = split_tfdataset_train_test(filename=samples)


def load_tokenizer():
	print("\n loading tokenizer")
	current_dir = dirname(realpath(__file__))
	# model_path = join(current_dir, model_folder, model_name)
	vocab_file = join(current_dir, model_dir, "vocab.txt")
	return FullTokenizer(vocab_file, do_lower_case=True)


# noinspection PyCompatibility
def process_raw_into_tf_records(samples, tokenizer):
	print("\n ")
	tf_records_file = samples + '.tfrecord'
	print(f"\n processing {samples} into {tf_records_file}...\n")
	sample_data = open(samples, 'r')
	records_writer = tf.io.TFRecordWriter(tf_records_file)
	for line in sample_data:
		words = line.split()
		content = tokenizer.tokenize(' '.join(words[1:]))
		token_ids = tokenizer.convert_tokens_to_ids(content)
		label = int(words[0] == 'ham')
		feature = {
			"token_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=token_ids)),
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
		}

		example = tf.train.Example(features=tf.train.Features(feature=feature))
		records_writer.write(example.SerializeToString())
	records_writer.close()
	sample_data.close()
	return tf_records_file


def load_tf_records(filename):
	"""
   convert TFRecord Pool objects back to TFRecordDataSet
   we compress and label in the process
   I am curious to see the shape
   """
	ds = tf.data.TFRecordDataset(filename)
	feature_description = {
		"token_ids": tf.io.VarLenFeature(tf.int64),
		"label": tf.io.FixedLenFeature([], tf.int64, default_value=-1)
	}

	def parse_protobuf(protobuf):
		example = tf.io.parse_single_example(protobuf, feature_description)
		token_ids, label = example["token_ids"], example["label"]
		token_ids = tf.compat.v1.sparse_tensor_to_dense(token_ids)
		return token_ids, label

	# < TFRecordDatasetV2 shapes: (), types: tf.string >
	return ds.map(parse_protobuf)


def brush_data(ds, tokenizer, pad_len=max_seq_len, batch_size=batch_size, buffer_size=buffer_size):
	pad_id, cls_id, sep_id = tokenizer.convert_tokens_to_ids(["[PAD]", "[CLS]", "[SEP]"])

	def padder(pad_len, trim_beginning=True):
		def set_padding(x, label):
			seq_len = pad_len - 2
			x = x[-seq_len:] if trim_beginning else x[:seq_len]
			x = tf.pad(x, [[0, seq_len - tf.shape(x)[-1]]], constant_values=pad_id)
			x = tf.concat([[cls_id], x, [sep_id]], axis=-1)
			return tf.reshape(x, (pad_len,)), tf.reshape(label, ())

		return set_padding

	# now, starting to pad/trim

	ds = ds.map(padder(pad_len=pad_len))

	ds = ds.shuffle(buffer_size=buffer_size, seed=4711, reshuffle_each_iteration=True)
	ds = ds.cache()
	ds = ds.repeat()

	ds = ds.batch(batch_size=batch_size, drop_remainder=True)
	return ds


# noinspection PyShadowingNames
def compile_model(max_seq_len=max_seq_len, adapter_size=adapter_size,
						batch_size=None, init_ckpt_file=None,
						init_bert_ckpt_file=bert_ckpt_file):
	"""

	:rtype: keras sequential model
	:param init_ckpt_file:
	:param max_seq_len:
	:param init_bert_ckpt_file:
	:param adapter_size:
	:type batch_size: integer
	"""
	# initializing Sequential model
	model = Sequential()
	# adding input_layer
	model.add(InputLayer(input_shape=(max_seq_len,), batch_size=batch_size, dtype="int32", name="input_ids"))
	# adding BERT layer
	bert_params = params_from_pretrained_ckpt(dirname(join(model_dir, 'bert_model.ckpt')))

	# create the bert layer
	bert_params.adapter_size = adapter_size
	bert_params.adapter_init_scale = 1e-5
	bert_layer = BertModelLayer.from_params(bert_params, name="bert")

	model.add(bert_layer)
	# adding temporal Dense, Normalization and Activation layers
	model.add(TimeDistributed(Dense(bert_params.hidden_size // 32)))
	model.add(TimeDistributed(LayerNormalization()))
	model.add(TimeDistributed(Activation("tanh")))
	model.add(Concat([
		Lambda(lambda x: tf.math.reduce_max(x, axis=1, keepdims=False)),
		GlobalAveragePooling1D()])
	)
	# dense_hidden_layer
	model.add(Dense(units=bert_params.hidden_size // 16))
	# normalization_layer
	model.add(LayerNormalization())
	# activation_layer
	model.add(Activation("tanh"))
	# dense_layer
	model.add(Dense(units=2))
	model.build(input_shape=(batch_size, max_seq_len))

	# freeze non-adapter-BERT layers for the case adapter_size is set
	bert_layer.apply_adapter_freeze()
	bert_layer.embeddings_layer.trainable = False  # True for unfreezing emb LayerNorms

	# apply global regularization on all trainable dense layers
	pf.utils.add_dense_layer_loss(model,
									kernel_regularizer=regularizers.l2(0.01),
									bias_regularizer=regularizers.l2(0.01))

	model.compile(optimizer=RAdam(),
					loss=SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[SparseCategoricalAccuracy(name="acc")])
	# load the pre-trained model weights (once the input_shape is known)
	if init_ckpt_file:
		print("Loading model weights from:", init_ckpt_file)
		model.load_weights(init_ckpt_file)
	elif init_bert_ckpt_file:
		print("Loading pre-trained BERT layer from:", init_bert_ckpt_file)
		load_stock_weights(bert_layer, init_bert_ckpt_file)

	return model


if __name__ == '__main__':
	tokenizer = load_tokenizer()

	train_tf_rec_file = process_raw_into_tf_records(samples=train_samples_file, tokenizer=tokenizer)

	train_tf_records = load_tf_records(filename=train_tf_rec_file)

	train_data = brush_data(ds=train_tf_records, tokenizer=tokenizer, batch_size=batch_size, buffer_size=buffer_size)
	# noinspection PyTypeChecker
	model = compile_model(max_seq_len=max_seq_len, adapter_size=adapter_size,
								batch_size=batch_size, init_bert_ckpt_file=bert_ckpt_file)
	model.summary()
	trained_ckpt_file = join(output_folder, 'checkpoints',
									'trained', 'sms_classification.ckpt', datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))

	if tf.io.gfile.exists(trained_ckpt_file):
		model.load_weights(trained_ckpt_file)
	else:
		log_dir = join(output_folder, "log", datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
		tensorboard_callback = TensorBoard(log_dir=log_dir)
		lr_scheduler = pf.utils.create_one_cycle_lr_scheduler(
				max_learn_rate=5e-3,  # experimental value!
				end_learn_rate=1e-6,  # experimental value!
				warmup_epoch_count=1,  # distort the initial values
				total_epoch_count=total_epoch_count)
		steps_per_epoch = expected_number_spam // batch_size
		steps_per_epoch = 2500 // batch_size
		early_stopping =  EarlyStopping(patience=10, restore_best_weights=True, monitor='loss')
		callbacks = [lr_scheduler, early_stopping, tensorboard_callback]
		history = model.fit(train_data, shuffle=True,
		 						epochs=total_epoch_count, steps_per_epoch=steps_per_epoch,
		 						callbacks=callbacks)
		#
		# model.save_weights(trained_ckpt_file, overwrite=True)

		# Evaluate the model
		test_tf_rec_file = process_raw_into_tf_records(samples=test_samples_file, tokenizer=tokenizer)
		test_tf_records = load_tf_records(filename=test_tf_rec_file)
		train_tf_rec_file = process_raw_into_tf_records(samples=train_samples_file, tokenizer=tokenizer)
		train_tf_records = load_tf_records(filename=train_tf_rec_file)

		test_data = brush_data(ds=test_tf_records, tokenizer=tokenizer, batch_size=batch_size, buffer_size=buffer_size)
		train_data = brush_data(ds=train_tf_records, tokenizer=tokenizer, batch_size=batch_size,
										buffer_size=buffer_size)

		_, train_acc = model.evaluate(train_data, steps=steps_per_epoch // batch_size)
		_, test_acc = model.evaluate(test_data, steps=steps_per_epoch // batch_size)
		print("train acc", train_acc)
		print(" test acc", test_acc)
