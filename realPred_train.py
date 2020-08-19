# In[4]:

import numpy as np
import tensorflow as tf
#from scipy import signal
# from disolve import remdiv
# modularized library import
import sys
import math
sys.path.append('/gdrive/My Drive/Colab Notebooks/Motion prediction')
#from calc_future_orientation import calc_future_orientation
#from ann_prediction import ann_prediction
from cap_prediction import cap_prediction
from crp_prediction import crp_prediction
from nop_prediction import nop_prediction
from solve_discontinuity import solve_discontinuity
#from convention_biosignal import convention_biosignal
#from convention_biosignal_quat import convention_biosignal_quat
from sklearn.externals import joblib

# In[5]:
#system_rate = 60
anticipation_time = 300
'''
#####   SYSTEM INITIALIZATION    #####
'''
tf.reset_default_graph()

tf.set_random_seed(2)
np.random.seed(2)

input_nm = 9
target_nm = 3

# Variables
TRAINED_MODEL_NAME = './best2_net3'
# Save it
scaler_file = "my_scaler2.save"
#joblib.dump(tempNorm, scaler_file) 
# Load it 
#tempNorm = joblib.load(scaler_file) 
# Reformat the input into TDNN format
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def pred(euler_data, gyro_data, time_data, x_seq_rt, timestamp_rt, eul_rt, gyro_rt, rt_counter, sess, model, DELAY_SIZE, y, x, midVel, avgVel):
# Neural network parameters
	#Get euler, gyro, and alfa one by one
	tempNorm = joblib.load(scaler_file) 
	euler_pred_onedata = solve_discontinuity(np.array(euler_data).reshape(1,-1))
	gyro_pred_onedata = np.array(gyro_data).reshape(1,-1)
	timestamp_rt[:-1] =timestamp_rt[1:]
	timestamp_rt[-1] = np.array(time_data).reshape(1,-1)
	gyro_rt[:-1] =gyro_rt[1:]
	gyro_rt[-1] = np.array(gyro_data).reshape(1,-1)
	eul_rt[:-1] =eul_rt[1:]
	eul_rt[-1] = np.array(euler_pred_onedata).reshape(1,-1)
	if (rt_counter==0):
		alfa_pred_onedata = np.array(np.zeros(shape=(1, 3), dtype=np.float))
		velocity_onedata = np.array(np.zeros(shape=(1, 3), dtype=np.float))
	else:
		alfa_pred_onedata = (np.diff(gyro_rt,axis=0)/np.diff(timestamp_rt, axis=0).reshape(-1, 1))*np.pi/180
		velocity_onedata = np.diff(eul_rt, axis=0)/np.diff(timestamp_rt, axis=0).reshape(-1, 1)

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

#		for j in range(0,3):
#			if (np.abs(velocity_onedata[:,j])<8):
#				y_sample[:,j] = tempNop[:,j]

		for j in range(0,3):
			xin = (np.abs(velocity_onedata[:,j])-midVel[j])/avgVel[j]
			alfa = sigmoid(xin)
			y_sample[:,j] = alfa*y_sample[:,j] + (1-alfa)*tempNop[:,j]

#		temp_rt = tempNorm.transform(temp_rt)
		x_seq_rt[:-1] = x_seq_rt[1:]
		x_seq_rt[-1] = temp_rt
		

	return y_sample, tempCap, tempCrp, tempNop, x_seq_rt