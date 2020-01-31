OS: MacOS Mojave 10.14.4
Python 3.7.6

#python modules
#not a full list, need to create conda env
tensorflow==2.1.0	
bert-for-tf2==0.13.4

python make_berty_up.py

We store SMSSpam in the date folder
"./data/SMSSpam"

The training looks as follows
_________________________________________________________________
Train for 46 steps
Epoch 00001: LearningRateScheduler reducing learning rate to 0.005.
Epoch 1/2
2020-01-26 23:53:55.499368: I tensorflow/core/profiler/lib/profiler_session.cc:225] Profiler session started.
46/46 [==============================] - 357s 8s/step - loss: 0.3938 - acc: 0.8696
Epoch 00002: LearningRateScheduler reducing learning rate to 0.005.
Epoch 2/2
46/46 [==============================] - 321s 7s/step - loss: 0.2327 - acc: 0.9443
Evaluating the model, we get
train_acc: 0.96875
test_acc: 0.96875

I fine-tune the pre-trained model ?freezing? BERT layer.
We compose BERT layer with Dense and Normalization and Activation
Total params: 109,077,146
Trainable params: 223,898
Non-trainable params: 108,853,248
I select butch_size*buffer_size= number_of_labels_spam
batch_size = 16
I assume we have 100 words at most
max_seq_len = 100 
Empirical quantity, looking at the size 5547 messages
adapter_size = 4
Two epochs looks enough
total_epoch_count  = 2
From the formula above
buffer_size = 47
%13,4 of 5574 messages
expected_number_spam = 747
