# In[4]:

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers
from scipy import signal
# from disolve import remdiv
# modularized library import
import sys
sys.path.append('/gdrive/My Drive/Colab Notebooks/Motion prediction')
from preparets import preparets
from train_test_split import train_test_split_tdnn
#from calc_future_orientation import calc_future_orientation
from eul2quat_bio import eul2quat_bio
from quat2eul_bio import quat2eul_bio
#from ann_prediction import ann_prediction
from cap_prediction import cap_prediction
from crp_prediction import crp_prediction
from nop_prediction import nop_prediction
from solve_discontinuity import solve_discontinuity
from rms import rms
#from convention_biosignal import convention_biosignal
#from convention_biosignal_quat import convention_biosignal_quat
from sklearn.externals import joblib

# In[5]:
system_rate = 60
anticipation_time = 300
'''
#####   SYSTEM INITIALIZATION    #####
'''
tf.reset_default_graph()

tf.set_random_seed(2)
np.random.seed(2)

# Create the time-shifted IMU data as the supervisor and assign the ann_feature as input
#anticipation_time = args.anticipation  # 앞 셀에서 정의함
anticipation_size = int(np.round(anticipation_time * system_rate / 1000))
print('anticipation size = ', anticipation_size)

input_nm = 9
target_nm = 3

# Neural network parameters
DELAY_SIZE = int(100 * (system_rate / 1000))  # 어떤 용도? 샘플 윈도우?
# Variables
TRAINED_MODEL_NAME = './best_net3'
# Save it
scaler_file = "my_scaler.save"
#joblib.dump(tempNorm, scaler_file) 
# Load it 
tempNorm = joblib.load(scaler_file) 
# Reformat the input into TDNN format
print('Anticipation time: {}ms\n'.format(anticipation_time))
# Reset the whole tensorflow graph
tf.reset_default_graph()
# Set up the placeholder to hold inputs and targets
x = tf.placeholder(dtype=tf.float32, shape=(None, DELAY_SIZE, input_nm))
t = tf.placeholder(dtype=tf.float32)


# In[20]:

# Define TDNN model
model = models.Sequential([
        layers.InputLayer(input_tensor=x, input_shape=(DELAY_SIZE, input_nm)),
        layers.Conv1D(27, DELAY_SIZE, activation=tf.nn.relu, input_shape=(DELAY_SIZE, input_nm), use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
        layers.Flatten(),
        layers.Dense(9, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(target_nm, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
        ])

# Get the output of the neural network model
y = model(x)

def pred(euler_data, gyro_data, time_data, x_seq_rt, timestamp_rt, gyro_rt, rt_counter, sess):

	#Get euler, gyro, and alfa one by one
	euler_pred_onedata = solve_discontinuity(np.array(euler_data).reshape(1,-1))
	gyro_pred_onedata = np.array(gyro_data).reshape(1,-1)
	timestamp_rt[:-1] =timestamp_rt[1:]
	timestamp_rt[-1] = np.array(time_data).reshape(1,-1)
	gyro_rt[:-1] =gyro_rt[1:]
	gyro_rt[-1] = np.array(gyro_data).reshape(1,-1)
	if (rt_counter==0):
		alfa_pred_onedata = np.array(np.zeros(shape=(1, 3), dtype=np.float))
	else:
		alfa_pred_onedata = (np.diff(gyro_rt,axis=0)/np.diff(timestamp_rt, axis=0).reshape(-1, 1))*np.pi/180

####################################Without SG Filter
	# Gather data until minimum delay is fulfilled
	if rt_counter < DELAY_SIZE:
		y_sample = [[0,0,0]]
		tempCap = [[0,0,0]]
		tempCrp = [[0,0,0]]
		tempNop = [[0,0,0]]
		temp_rt = np.column_stack([euler_pred_onedata, gyro_pred_onedata, alfa_pred_onedata])
#		temp_rt = tempNorm.transform(temp_rt)

		x_seq_rt[:-1] = x_seq_rt[1:]
		x_seq_rt[-1] = temp_rt
#		if (rt_counter == sliding_window_size):
#			seq_df = pd.DataFrame(x_seq_rt)
#			seq_df = seq_df.rolling(sliding_window_size, min_periods=1).mean()
#			temp_seq_rt = np.array(seq_df)
#			xd_seq_rt[:-1] = xd_seq_rt[1:]
#			xd_seq_rt[-1] = temp_seq_rt[-1]
#		else:
#			xd_seq_rt[:-1] = xd_seq_rt[1:]
#			xd_seq_rt[-1] = temp_rt
	else:
		model.load_weights(TRAINED_MODEL_NAME)
		xdd_seq_rt = tempNorm.transform(x_seq_rt)
		y_sample = sess.run(y, feed_dict={x:xdd_seq_rt.reshape(1,DELAY_SIZE,input_nm)})

		#Get temp cap crp and nop
		tempCap = cap_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, alfa_pred_onedata, anticipation_time)
		tempCrp = crp_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, anticipation_time)
		tempNop = nop_prediction(euler_pred_onedata, anticipation_time)
		temp_rt = np.column_stack([euler_pred_onedata, gyro_pred_onedata, alfa_pred_onedata])

#		temp_rt = tempNorm.transform(temp_rt)
		x_seq_rt[:-1] = x_seq_rt[1:]
		x_seq_rt[-1] = temp_rt
		

	return y_sample, tempCap, tempCrp, tempNop, x_seq_rt