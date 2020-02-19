Environment
OS: MacOS Mojave 10.14.4
Python 3.7.6
Requirements
conda 4.7.12
conda env create --name envname --file=environments.yml
Python 3.7.6

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

