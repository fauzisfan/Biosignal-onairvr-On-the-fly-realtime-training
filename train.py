# In[4]:

import numpy as np
import pandas as pd
import tensorflow as tf
#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers
# from disolve import remdiv
# modularized library import
#import sys
#sys.path.append('/gdrive/My Drive/Colab Notebooks/Motion prediction')
from preparets import preparets
#from train_test_split import train_test_split_tdnn
#from calc_future_orientation import calc_future_orientation
#from ann_prediction import ann_prediction
from solve_discontinuity import solve_discontinuity
#from convention_biosignal import convention_biosignal
#from convention_biosignal_quat import convention_biosignal_quat   
from sklearn.externals import joblib

#class train():
def sett(self, x, y , z): 
    self.flag = x
    self.model = y
    self.DELAY_SIZE = z
    
def get(self):
    return  self.flag, self.model, self.DELAY_SIZE
	
        
def train_nn(eul_train, gyro_train, time_stamp):
    anticipation_time = 300
    np.random.seed(2)
    train_gyro_data = gyro_train
    train_eule_data = eul_train
    train_time_data = time_stamp
    train_data_id = len(eul_train)

    '''
    #####   데이터 로드    #####
    '''
    print('Training data preprocessing is started...')
    
    
    # Remove zero data from collected training data
    system_rate = round((train_data_id+1)/float(np.max(train_time_data) - train_time_data[0]))
#    train_gyro_data = train_gyro_data* 180/ np.pi
    train_alfa_data = np.diff(train_gyro_data, axis=0)/np.diff(train_time_data, axis=0)
    train_alfa_data = np.row_stack([np.zeros(shape=(1, train_alfa_data.shape[1]), dtype=np.float), train_alfa_data])
    train_alfa_data = train_alfa_data * np.pi / 180
    
    """Velocity data"""
    anticipation_size = int(anticipation_time*system_rate/1000)
    train_velocity_data = np.diff(train_eule_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
    train_velocity_data = np.row_stack([np.zeros(shape=(1, train_velocity_data.shape[1]), dtype=np.float), train_velocity_data])
    velocity_o = train_velocity_data[:-anticipation_size]
    
    train_eule_data = solve_discontinuity(train_eule_data)
    
    # Create data frame of all features and smoothing
#    sliding_window_time = 100
#    sliding_window_size = int(np.round(sliding_window_time * system_rate / 1000))
    
    ann_feature = np.column_stack([train_eule_data, 
                                   train_gyro_data, 
                                   train_alfa_data, 
    #                               train_magn_data,
                                   ])
        
    feature_name = ['pitch', 'roll', 'yaw', 
                    'gX', 'gY', 'gZ', 
                    'aX', 'aY', 'aZ', 
    #                'mX', 'mY', 'mZ', 
    #                'EMG1', 'EMG2', 'EMG3', 'EMG4',
                    ]
    
    ann_feature_df = pd.DataFrame(ann_feature, columns=feature_name)
    #ann_feature_df = ann_feature_df.rolling(sliding_window_size, min_periods=1).mean()
    
    # Create the time-shifted IMU data as the supervisor and assign the ann_feature as input
    #anticipation_time = args.anticipation  # 앞 셀에서 정의함
    anticipation_size = int(np.round(anticipation_time * system_rate / 1000))
    print('anticipation size = ', anticipation_size)
    
    spv_name = ['pitch', 'roll', 'yaw']
    target_series_df = ann_feature_df[spv_name].iloc[anticipation_size::].reset_index(drop=True)
    input_series_df = ann_feature_df.iloc[:-anticipation_size].reset_index(drop=True)
    
    input_nm = len(input_series_df.columns)
    target_nm = len(target_series_df.columns)
    
    # In[17]:
    
    '''
    #####   NN 입력 데이터 준비    #####
    '''
    # Neural network parameters
    DELAY_SIZE = int(100*system_rate/1000)
#    TEST_SIZE = 0.1
    # Variables
    TRAINED_MODEL_NAME = './best2_net3'
    
    # Import datasets
    input_series = np.array(input_series_df)
    target_series = np.array(target_series_df)
    normalizer = preprocessing.StandardScaler()
    #Get normalized based data on the 
    tempNorm = normalizer.fit(input_series)
    # Save it
    scaler_file = "my_scaler2.save"
    joblib.dump(tempNorm, scaler_file) 
    #Normalizer used on input series
    input_norm = tempNorm.transform(input_series)
    
    # Reformat the input into TDNN format
    x_seq, t_seq = preparets(input_norm, target_series, DELAY_SIZE)
    print('Anticipation time: {}ms\n'.format(anticipation_time))
    # Reset the whole tensorflow graph

#    x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, t_seq, TEST_SIZE)
    # Set up the placeholder to hold inputs and targets
    x = tf.placeholder(dtype=tf.float32, shape=(None, DELAY_SIZE, input_nm))
    t = tf.placeholder(dtype=tf.float32)
    
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
    
    # Define the loss
    loss = tf.reduce_mean(tf.square(t-y))
    
    total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, t)))
    R_squared = tf.subtract(1.0, tf.div(unexplained_error, total_error))

    n_epochs=1000
    learning_rate = 0.01
    optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
    # Start tensorflow session
    with tf.Session() as sess:
        # Initiate variable values
        sess.run(tf.global_variables_initializer())
        
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", R_squared)
        merged_summary = tf.summary.merge_all()
        
    
        for epoch in range (n_epochs):
            summary,_= sess.run([merged_summary,training_op], feed_dict={x:x_seq, t:t_seq})
    
        model.save_weights(TRAINED_MODEL_NAME)
        
    flag = 1
    
    midVel = np.nanpercentile(np.abs(velocity_o),50, axis = 0)
    avgVel = np.nanmean(np.abs(velocity_o), axis=0)
#        self.sett(flag, model, DELAY_SIZE)
    
    return flag, model, DELAY_SIZE, midVel, avgVel