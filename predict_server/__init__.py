#import math
#import asyncio
#import time
#import zmq
#import cbor2
#import tensorflow as tf
#from abc import abstractmethod, ABCMeta
#from zmq.asyncio import Context, Poller
#from realPred import pred
#
#from ._types import MotionData, PredictedData
#from ._writer import PredictionOutputWriter, PerfMetricWriter
#import numpy as np
#
#class PredictModule(metaclass=ABCMeta):
#    @abstractmethod
#    def predict(self, motion_data):
#        pass
#
#    @abstractmethod
#    def feedback_received(self, feedback):
#        pass
#
#    def make_camera_projection(self, motion_data, overfilling):
#        return [
#            math.tan(math.atan(motion_data.camera_projection[0]) - overfilling[0]),
#            math.tan(math.atan(motion_data.camera_projection[1]) + overfilling[1]),
#            math.tan(math.atan(motion_data.camera_projection[2]) + overfilling[2]),
#            math.tan(math.atan(motion_data.camera_projection[3]) - overfilling[3])
#        ]        
#
#class Geek: 
#    def __init__(self, val = 0): 
#         self._val = val 
#      
#    # getter method 
#    def get_age(self): 
#        return self._val 
#      
#    # setter method 
#    def set_age(self, x): 
#        self._val = x
#		
#class MotionPredictServer:
#    ann_pred_rt = np.zeros((0,3), dtype= float)
#    cap_pred_rt = np.zeros((0,3), dtype= float)
#    crp_pred_rt = np.zeros((0,3), dtype= float)
#    nop_pred_rt = np.zeros((0,3), dtype= float)
#    ori_rt = np.zeros((0,3), dtype= float)
#    
#    def __init__(self, module, port_input, port_feedback,
#                 prediction_output, metric_output):
#        self.module = module
#        self.port_input = port_input
#        self.port_feedback = port_feedback
#        self.feedbacks = {}
#		
#        self.prediction_output = PredictionOutputWriter(
#            prediction_output
#        ) if prediction_output is not None else None
#        
#        self.metric_writer = PerfMetricWriter(
#            metric_output
#        ) if metric_output is not None else None
#				
#    def get_euler(self): 
#        return self.input_euler 
#	
#    def set_euler(self, x): 
#        self.input_euler = x
#		
#    def run(self):
#        context = Context.instance()
#        self.event_loop = asyncio.get_event_loop()
#
#        self.event_loop.run_until_complete(self.loop(context))
#
#    def shutdown(self):
#        self.event_loop.close()
#        
#        ann_mae = np.nanmean(np.abs(self.ann_pred_rt[:-(18+18)] - self.ori_rt[18:-18]), axis=0)
#        crp_mae = np.nanmean(np.abs(self.crp_pred_rt[:-(18+18)] - self.ori_rt[18:-18]), axis=0)
#        cap_mae = np.nanmean(np.abs(self.cap_pred_rt[:-(18+18)] - self.ori_rt[18:-18]), axis=0)
#        nop_mae = np.nanmean(np.abs(self.nop_pred_rt[:-(18+18)] - self.ori_rt[18:-18]), axis=0)
#        
#        final_ann_rt_99 = np.nanpercentile(np.abs(self.ann_pred_rt[:-(18+18)] - self.ori_rt[18:-18]),99, axis = 0)
#        final_crp_rt_99 = np.nanpercentile(np.abs(self.crp_pred_rt[:-(18+18)] - self.ori_rt[18:-18]),99, axis = 0)
#        final_cap_rt_99 = np.nanpercentile(np.abs(self.cap_pred_rt[:-(18+18)] - self.ori_rt[18:-18]),99, axis = 0)
#        final_nop_rt_99 = np.nanpercentile(np.abs(self.nop_pred_rt[:-(18+18)] - self.ori_rt[18:-18]),99, axis = 0)
#        
#        print('\nFinal Result of MAE:')
#        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae[0], ann_mae[1], ann_mae[2]))
#        print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(crp_mae[0], crp_mae[1], crp_mae[2]))
#        print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(cap_mae[0], cap_mae[1], cap_mae[2]))
#        print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(nop_mae[0], nop_mae[1], nop_mae[2]))
#        
#        print('\nFinal Result of 99% Error:')
#        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_ann_rt_99[0], final_ann_rt_99[1], final_ann_rt_99[2]))
#        print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_crp_rt_99[0], final_crp_rt_99[1], final_crp_rt_99[2]))
#        print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_cap_rt_99[0], final_cap_rt_99[1], final_cap_rt_99[2]))
#        print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_nop_rt_99[0], final_nop_rt_99[1], final_nop_rt_99[2]))
#        
#        if self.prediction_output is not None:
#            self.prediction_output.close()
#
#        if self.metric_writer is not None:
#            self.metric_writer.close()
#
#    async def loop(self, context):
#        motion_raw_input = context.socket(zmq.PULL)
#        motion_raw_input.bind("tcp://*:" + str(self.port_input))
#
#        motion_predicted_output = context.socket(zmq.PUSH)
#        motion_predicted_output.bind("tcp://*:" + str(self.port_input + 1))
#
#        feedback_recv = context.socket(zmq.PULL)
#        feedback_recv.bind("tcp://*:" + str(self.port_feedback))
#
#        poller = Poller()
#        poller.register(motion_raw_input, zmq.POLLIN)
#        poller.register(feedback_recv, zmq.POLLIN)
#        rt_counter = 0
#
#        x_seq_rt = np.zeros(shape=(6, 9))
#        timestamp_rt = np.zeros(shape=(2, 1), dtype= float)
#        gyro_rt = np.zeros(shape=(2, 3), dtype= float)
##        xd_seq_rt = np.zeros(shape=(3, 9))
##        import RealTimePlot as rtp
##        y_plot = np.zeros(5)
#        with tf.Session() as sess:
#            while True:
#                events = await poller.poll(100)
#                if motion_raw_input in dict(events):
#                    frame = await motion_raw_input.recv(0, False)
#                    motion_data = MotionData.from_bytes(frame.bytes)
#    
#                    self.start_prediction(motion_data.timestamp)
#    
#                    prediction_time, predicted_orientation, predicted_projection = \
#                        self.module.predict(motion_data)
#    
##                    Output StreamLine
##                    self.set_euler(self.prediction_output.quat_to_euler(motion_data.orientation))
#                    masukan_euler = self.prediction_output.quat_to_euler(motion_data.orientation)#self.get_euler()
#                    input_euler = np.array([masukan_euler[1],masukan_euler[2],masukan_euler[0]])*180/np.pi
#                    input_gyro = np.array(motion_data.angular_velocities)*180/np.pi
#                    input_time = motion_data.timestamp/ 705600000
#                    
#                    ann, cap, crp, nop, x_seq_rt = pred(input_euler, input_gyro, input_time, x_seq_rt, timestamp_rt, gyro_rt, rt_counter, sess)
#                    print('\nInput [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(input_euler[0],input_euler[1],input_euler[2]))
#                    print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann[0][0],ann[0][1],ann[0][2]))
##                    print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(cap[0][0],cap[0][1],cap[0][2]))
##                    print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(crp[0][0],crp[0][1],crp[0][2]))
##                    print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(nop[0][0],nop[0][1],nop[0][2]))
#                    
#
#                    if (rt_counter >= 1020):# get the data after 17 seconds (60 FPS is assumed)
#                        self.ori_rt = np.concatenate((self.ori_rt, np.array(input_euler).reshape(1,-1)), axis =0)
#                        self.ann_pred_rt = np.concatenate((self.ann_pred_rt, ann), axis =0)
#                        self.crp_pred_rt = np.concatenate((self.crp_pred_rt, crp), axis =0)
#                        self.cap_pred_rt = np.concatenate((self.cap_pred_rt, cap), axis =0)
#                        self.nop_pred_rt = np.concatenate((self.nop_pred_rt, nop), axis =0)
##                        pastTime = rt_counter-18
#                        
#                    #visualization of realtime
##                    if (rt_counter >= 1038):
##                        y_plot[0] = ori_rt[rt_counter][2]
##                        y_plot[1] = crp_pred_rt[pastTime][2]
##                        y_plot[2] = cap_pred_rt[pastTime][2]
##                        y_plot[3] = nop_pred_rt[pastTime][2]
##                        y_plot[4] = ann_pred_rt[pastTime][2]
##                        rtp.RealTimePlot(float(rt_counter/60), y_plot)
#                    rt_counter += 1
#    
#                    predicted_data = PredictedData(motion_data.timestamp,
#                                                   prediction_time,
#                                                   predicted_orientation,
#                                                   predicted_projection)
#                    
#                    self.end_prediction(motion_data.timestamp)
#                
#                    motion_predicted_output.send(predicted_data.pack())
#    
#                    if self.prediction_output is not None:
#                        self.prediction_output.write(motion_data, predicted_data)
#                    
#                if feedback_recv in dict(events):
#                    feedback = await feedback_recv.recv()
#                    self.merge_feedback(cbor2.loads(feedback))
#
#
#    # process feedbacks
#    def start_prediction(self, session):
#        assert(session not in self.feedbacks)
#        self.feedbacks[session] = {
#            'srcmask': 0,
#            'startPrediction': time.clock()
#        }
#
#    def end_prediction(self, session):
#        self.feedbacks[session]['stopPrediction'] = time.clock()
#
#    def merge_feedback(self, feedback):
#        if not all(key in feedback for key in ('session', 'source')):
#            return
#
#        if not feedback['session'] in self.feedbacks:
#            return
#        
#        session = feedback['session']
#        entry = self.feedbacks[session]
#        
#        if feedback['source'] == 'acli':
#            entry['srcmask'] |= 0b01
#        elif feedback['source'] == 'asrv':
#            entry['srcmask'] |= 0b10
#        else:
#            return
#
#        del feedback['source']
#        self.feedbacks[session] = {**entry, **feedback}
#
#        if entry['srcmask'] == 0b11:
#            if self.metric_writer is not None:
#                self.metric_writer.write_metric(self.feedbacks[session])
#
#            self.module.feedback_received(self.feedbacks[session])
#                
#            self.feedbacks = {
#                s: self.feedbacks[s] for s in self.feedbacks if s > session
#            }

#''' Coba on-fly- Training'''
## -*- coding: utf-8 -*-
#"""
#Created on Fri Feb 28 15:19:23 2020
#
#@author: isfan
#"""
#
#import math
#import asyncio
#import time
#import zmq
#import cbor2
#import tensorflow as tf
#from abc import abstractmethod, ABCMeta
#from zmq.asyncio import Context, Poller
#from realPred_train import pred
#from train import train_nn
##from train import train_nn
#from ._types import MotionData, PredictedData
#from ._writer import PredictionOutputWriter, PerfMetricWriter
#import numpy as np
##import _thread
#
#class PredictModule(metaclass=ABCMeta):
#    @abstractmethod
#    def predict(self, motion_data):
#        pass
#
#    @abstractmethod
#    def feedback_received(self, feedback):
#        pass
#
#    def make_camera_projection(self, motion_data, overfilling):
#        return [
#            math.tan(math.atan(motion_data.camera_projection[0]) - overfilling[0]),
#            math.tan(math.atan(motion_data.camera_projection[1]) + overfilling[1]),
#            math.tan(math.atan(motion_data.camera_projection[2]) + overfilling[2]),
#            math.tan(math.atan(motion_data.camera_projection[3]) - overfilling[3])
#        ]        
#
#class Geek: 
#    def __init__(self, val = 0): 
#         self._val = val 
#      
#    # getter method 
#    def get_age(self): 
#        return self._val 
#      
#    # setter method 
#    def set_age(self, x): 
#        self._val = x
#		
#class MotionPredictServer:
#    ann_pred_rt = np.zeros((0,3), dtype= float)
#    cap_pred_rt = np.zeros((0,3), dtype= float)
#    crp_pred_rt = np.zeros((0,3), dtype= float)
#    nop_pred_rt = np.zeros((0,3), dtype= float)
#    ori_rt = np.zeros((0,3), dtype= float)
#    
#    def __init__(self, module, port_input, port_feedback,
#                 prediction_output, metric_output):
#        self.module = module
#        self.port_input = port_input
#        self.port_feedback = port_feedback
#        self.feedbacks = {}
#		
#        self.prediction_output = PredictionOutputWriter(
#            prediction_output
#        ) if prediction_output is not None else None
#        
#        self.metric_writer = PerfMetricWriter(
#            metric_output
#        ) if metric_output is not None else None
#		
#    def run(self):
#        context = Context.instance()
#        self.event_loop = asyncio.get_event_loop()
#
#        self.event_loop.run_until_complete(self.loop(context))
#
#    def shutdown(self):
#        self.event_loop.close()
#        
#        ann_mae = np.nanmean(np.abs(self.ann_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]), axis=0)
#        crp_mae = np.nanmean(np.abs(self.crp_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]), axis=0)
#        cap_mae = np.nanmean(np.abs(self.cap_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]), axis=0)
#        nop_mae = np.nanmean(np.abs(self.nop_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]), axis=0)
#        
#        final_ann_rt_99 = np.nanpercentile(np.abs(self.ann_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]),99, axis = 0)
#        final_crp_rt_99 = np.nanpercentile(np.abs(self.crp_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]),99, axis = 0)
#        final_cap_rt_99 = np.nanpercentile(np.abs(self.cap_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]),99, axis = 0)
#        final_nop_rt_99 = np.nanpercentile(np.abs(self.nop_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]),99, axis = 0)
#        
#        print('\nFinal Result of MAE:')
#        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae[0], ann_mae[1], ann_mae[2]))
#        print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(crp_mae[0], crp_mae[1], crp_mae[2]))
#        print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(cap_mae[0], cap_mae[1], cap_mae[2]))
#        print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(nop_mae[0], nop_mae[1], nop_mae[2]))
#        
#        print('\nFinal Result of 99% Error:')
#        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_ann_rt_99[0], final_ann_rt_99[1], final_ann_rt_99[2]))
#        print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_crp_rt_99[0], final_crp_rt_99[1], final_crp_rt_99[2]))
#        print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_cap_rt_99[0], final_cap_rt_99[1], final_cap_rt_99[2]))
#        print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_nop_rt_99[0], final_nop_rt_99[1], final_nop_rt_99[2]))
#        
#        if self.prediction_output is not None:
#            self.prediction_output.close()
#
#        if self.metric_writer is not None:
#            self.metric_writer.close()
#
#    async def loop(self, context):
#        motion_raw_input = context.socket(zmq.PULL)
#        motion_raw_input.bind("tcp://*:" + str(self.port_input))
#
#        motion_predicted_output = context.socket(zmq.PUSH)
#        motion_predicted_output.bind("tcp://*:" + str(self.port_input + 1))
#
#        feedback_recv = context.socket(zmq.PULL)
#        feedback_recv.bind("tcp://*:" + str(self.port_feedback))
#
#        poller = Poller()
#        poller.register(motion_raw_input, zmq.POLLIN)
#        poller.register(feedback_recv, zmq.POLLIN)
#        rt_counter = 0
#
#        x_seq_rt = np.zeros(shape=(6, 9))
#        timestamp_rt = np.zeros(shape=(2, 1), dtype= float)
#        gyro_rt = np.zeros(shape=(2, 3), dtype= float)
#        eul_rt = np.zeros(shape=(2, 3), dtype= float)
##        xd_seq_rt = np.zeros(shape=(3, 9))
##        import RealTimePlot as rtp
##        y_plot = np.zeros(5)
#        
#        flag = 0
#        trainTime = 3*3600 #5 minutes
#        train_counter = 0
#        eul_train = np.zeros((0,3), dtype= float)
#        gyro_train = np.zeros((0,3), dtype= float)
#        time_stamp = np.zeros((0,1), dtype= float)
#        input_nm = 9
##        trainInit = train.train()
#        with tf.Session() as sess:
#            while True:
#                events = await poller.poll(100)
#                if motion_raw_input in dict(events):
#                    frame = await motion_raw_input.recv(0, False)
#                    motion_data = MotionData.from_bytes(frame.bytes)
#    
#                    self.start_prediction(motion_data.timestamp)
#    
#                    prediction_time, predicted_orientation, predicted_projection = \
#                        self.module.predict(motion_data)
#    
##                    Output StreamLine
##                    self.set_euler(self.prediction_output.quat_to_euler(motion_data.orientation))
#                    masukan_euler = self.prediction_output.quat_to_euler(motion_data.orientation)#self.get_euler()
#                    input_euler = np.array([masukan_euler[1],masukan_euler[2],masukan_euler[0]])*180/np.pi
#                    input_gyro = np.array(motion_data.angular_velocities)*180/np.pi
#                    input_time = motion_data.timestamp/ 705600000
#                    
#                    if (flag == 0 and train_counter >= 1020):
#                        eul_train = np.concatenate((eul_train, np.array(input_euler).reshape(1,-1)), axis =0)
#                        gyro_train = np.concatenate((gyro_train, np.array(input_gyro).reshape(1,-1)), axis =0)
#                        time_stamp = np.concatenate((time_stamp, np.array(input_time).reshape(1,-1)), axis =0)
#                        print(train_counter)
#                        if (train_counter == 1020+trainTime):
##                            _thread.start_new_thread(trainInit.train_nn(eul_train, gyro_train, time_stamp), ())
#                            flag, model, DELAY_SIZE, midVel, avgVel = train_nn(eul_train, gyro_train, time_stamp)#trainInit.get()
#                            print(midVel)
#                            print(avgVel)
#                            x = tf.placeholder(dtype=tf.float32, shape=(None, DELAY_SIZE, input_nm))
#                            y = model(x)
#                    elif (flag == 1):# get the data after 17 seconds (60 FPS is assumed)
#                        ann, cap, crp, nop, x_seq_rt = pred(input_euler, input_gyro, input_time, x_seq_rt, timestamp_rt, eul_rt, gyro_rt, rt_counter, sess, model, DELAY_SIZE, y, x, midVel, avgVel)
#                        print('\nInput [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(input_euler[0],input_euler[1],input_euler[2]))
#                        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann[0][0],ann[0][1],ann[0][2]))
#                        self.ori_rt = np.concatenate((self.ori_rt, np.array(input_euler).reshape(1,-1)), axis =0)
#                        self.ann_pred_rt = np.concatenate((self.ann_pred_rt, ann), axis =0)
#                        self.crp_pred_rt = np.concatenate((self.crp_pred_rt, crp), axis =0)
#                        self.cap_pred_rt = np.concatenate((self.cap_pred_rt, cap), axis =0)
#                        self.nop_pred_rt = np.concatenate((self.nop_pred_rt, nop), axis =0)
#                        rt_counter += 1
#                    train_counter += 1
#                    #visualization of realtime
#    #                    if (rt_counter >= 1038):
#    #                        y_plot[0] = ori_rt[rt_counter][2]
#    #                        y_plot[1] = crp_pred_rt[pastTime][2]
#    #                        y_plot[2] = cap_pred_rt[pastTime][2]
#    #                        y_plot[3] = nop_pred_rt[pastTime][2]
#    #                        y_plot[4] = ann_pred_rt[pastTime][2]
#    #                        rtp.RealTimePlot(float(rt_counter/60), y_plot)
#    
#                    predicted_data = PredictedData(motion_data.timestamp,
#                                                   prediction_time,
#                                                   predicted_orientation,
#                                                   predicted_projection)
#                    
#                    self.end_prediction(motion_data.timestamp)
#                
#                    motion_predicted_output.send(predicted_data.pack())
#    
#                    if self.prediction_output is not None:
#                        self.prediction_output.write(motion_data, predicted_data)
#                    
#                if feedback_recv in dict(events):
#                    feedback = await feedback_recv.recv()
#                    self.merge_feedback(cbor2.loads(feedback))
#
#
#    # process feedbacks
#    def start_prediction(self, session):
#        assert(session not in self.feedbacks)
#        self.feedbacks[session] = {
#            'srcmask': 0,
#            'startPrediction': time.clock()
#        }
#
#    def end_prediction(self, session):
#        self.feedbacks[session]['stopPrediction'] = time.clock()
#
#    def merge_feedback(self, feedback):
#        if not all(key in feedback for key in ('session', 'source')):
#            return
#
#        if not feedback['session'] in self.feedbacks:
#            return
#        
#        session = feedback['session']
#        entry = self.feedbacks[session]
#        
#        if feedback['source'] == 'acli':
#            entry['srcmask'] |= 0b01
#        elif feedback['source'] == 'asrv':
#            entry['srcmask'] |= 0b10
#        else:
#            return
#
#        del feedback['source']
#        self.feedbacks[session] = {**entry, **feedback}
#
#        if entry['srcmask'] == 0b11:
#            if self.metric_writer is not None:
#                self.metric_writer.write_metric(self.feedbacks[session])
#
#            self.module.feedback_received(self.feedbacks[session])
#                
#            self.feedbacks = {
#                s: self.feedbacks[s] for s in self.feedbacks if s > session
#            }

''' Coba MultiProcessor- Training'''
# -*- coding: utf-8 -*-
"""

@author: isfan
"""

import math
import asyncio
import time
import zmq
import cbor2
import tensorflow as tf
from abc import abstractmethod, ABCMeta
from zmq.asyncio import Context, Poller
from realPred_train import pred
from train import train_nn
from ._types import MotionData, PredictedData
from ._writer import PredictionOutputWriter, PerfMetricWriter
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers
from preparets import preparets
from solve_discontinuity import solve_discontinuity
from sklearn.externals import joblib

import multiprocessing

    
flag = model = DELAY_SIZE = midVel = avgVel = None
    
class PredictModule(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, motion_data):
        pass

    @abstractmethod
    def feedback_received(self, feedback):
        pass

    def make_camera_projection(self, motion_data, overfilling):
        return [
            math.tan(math.atan(motion_data.camera_projection[0]) - overfilling[0]),
            math.tan(math.atan(motion_data.camera_projection[1]) + overfilling[1]),
            math.tan(math.atan(motion_data.camera_projection[2]) + overfilling[2]),
            math.tan(math.atan(motion_data.camera_projection[3]) - overfilling[3])
        ]        

class Geek: 
    def __init__(self, val = 0): 
         self._val = val 
      
    # getter method 
    def get_age(self): 
        return self._val 
      
    # setter method 
    def set_age(self, x): 
        self._val = x
		
class MotionPredictServer:
    ann_pred_rt = np.zeros((0,3), dtype= float)
    cap_pred_rt = np.zeros((0,3), dtype= float)
    crp_pred_rt = np.zeros((0,3), dtype= float)
    nop_pred_rt = np.zeros((0,3), dtype= float)
    ori_rt = np.zeros((0,3), dtype= float)
    
    flag = midVel = avgVel = None
    
    def __init__(self, module, port_input, port_feedback,
                 prediction_output, metric_output):
        self.module = module
        self.port_input = port_input
        self.port_feedback = port_feedback
        self.feedbacks = {}
		
        self.prediction_output = PredictionOutputWriter(
            prediction_output
        ) if prediction_output is not None else None
        
        self.metric_writer = PerfMetricWriter(
            metric_output
        ) if metric_output is not None else None
		
    def run(self):
        context = Context.instance()
        self.event_loop = asyncio.get_event_loop()

        self.event_loop.run_until_complete(self.loop(context))

    def shutdown(self):
        self.event_loop.close()
        
        ann_mae = np.nanmean(np.abs(self.ann_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]), axis=0)
        crp_mae = np.nanmean(np.abs(self.crp_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]), axis=0)
        cap_mae = np.nanmean(np.abs(self.cap_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]), axis=0)
        nop_mae = np.nanmean(np.abs(self.nop_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]), axis=0)
        
        final_ann_rt_99 = np.nanpercentile(np.abs(self.ann_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]),99, axis = 0)
        final_crp_rt_99 = np.nanpercentile(np.abs(self.crp_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]),99, axis = 0)
        final_cap_rt_99 = np.nanpercentile(np.abs(self.cap_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]),99, axis = 0)
        final_nop_rt_99 = np.nanpercentile(np.abs(self.nop_pred_rt[6:-(18+18)] - self.ori_rt[18+6:-18]),99, axis = 0)
        
        print('\nFinal Result of MAE:')
        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae[0], ann_mae[1], ann_mae[2]))
        print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(crp_mae[0], crp_mae[1], crp_mae[2]))
        print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(cap_mae[0], cap_mae[1], cap_mae[2]))
        print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(nop_mae[0], nop_mae[1], nop_mae[2]))
        
        print('\nFinal Result of 99% Error:')
        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_ann_rt_99[0], final_ann_rt_99[1], final_ann_rt_99[2]))
        print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_crp_rt_99[0], final_crp_rt_99[1], final_crp_rt_99[2]))
        print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_cap_rt_99[0], final_cap_rt_99[1], final_cap_rt_99[2]))
        print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_nop_rt_99[0], final_nop_rt_99[1], final_nop_rt_99[2]))
        
        if self.prediction_output is not None:
            self.prediction_output.close()

        if self.metric_writer is not None:
            self.metric_writer.close()

    async def loop(self, context):
        motion_raw_input = context.socket(zmq.PULL)
        motion_raw_input.bind("tcp://*:" + str(self.port_input))

        motion_predicted_output = context.socket(zmq.PUSH)
        motion_predicted_output.bind("tcp://*:" + str(self.port_input + 1))

        feedback_recv = context.socket(zmq.PULL)
        feedback_recv.bind("tcp://*:" + str(self.port_feedback))

        poller = Poller()
        poller.register(motion_raw_input, zmq.POLLIN)
        poller.register(feedback_recv, zmq.POLLIN)
        rt_counter = 0

        x_seq_rt = np.zeros(shape=(6, 9))
        timestamp_rt = np.zeros(shape=(2, 1), dtype= float)
        gyro_rt = np.zeros(shape=(2, 3), dtype= float)
        eul_rt = np.zeros(shape=(2, 3), dtype= float)
#        xd_seq_rt = np.zeros(shape=(3, 9))
#        import RealTimePlot as rtp
#        y_plot = np.zeros(5)
        
        flag = 0
        trainTime = 1*3600 #5 minutes
        train_counter = 0
        eul_train = np.zeros((0,3), dtype= float)
        gyro_train = np.zeros((0,3), dtype= float)
        time_stamp = np.zeros((0,1), dtype= float)
        input_nm = 9
#        trainInit = train.train()
        with tf.Session() as sess:
            while True:
                events = await poller.poll(100)
                if motion_raw_input in dict(events):
                    frame = await motion_raw_input.recv(0, False)
                    motion_data = MotionData.from_bytes(frame.bytes)
    
                    self.start_prediction(motion_data.timestamp)
    
                    prediction_time, predicted_orientation, predicted_projection = \
                        self.module.predict(motion_data)
    
#                    Output StreamLine
#                    self.set_euler(self.prediction_output.quat_to_euler(motion_data.orientation))
                    masukan_euler = self.prediction_output.quat_to_euler(motion_data.orientation)#self.get_euler()
                    input_euler = np.array([masukan_euler[1],masukan_euler[2],masukan_euler[0]])*180/np.pi
                    input_gyro = np.array(motion_data.angular_velocities)*180/np.pi
                    input_time = motion_data.timestamp/ 705600000
                    
                    if (flag == 0 and train_counter >= 1020):
                        eul_train = np.concatenate((eul_train, np.array(input_euler).reshape(1,-1)), axis =0)
                        gyro_train = np.concatenate((gyro_train, np.array(input_gyro).reshape(1,-1)), axis =0)
                        time_stamp = np.concatenate((time_stamp, np.array(input_time).reshape(1,-1)), axis =0)
                        print(train_counter)
                        if (train_counter == 1020+trainTime):
                            flag, model, DELAY_SIZE, midVel, avgVel = train_nn(eul_train, gyro_train, time_stamp)
#                            trainProcessing = multiprocessing.Process(target = MotionPredictServer.train_nn, args =(eul_train, gyro_train, time_stamp))
#                            trainProcessing.start()
#                            trainProcessing.join()
                    elif (flag == 1):# get the data after 17 seconds (60 FPS is assumed)
                        if (rt_counter == 0) :
                            x = tf.placeholder(dtype=tf.float32, shape=(None, DELAY_SIZE, input_nm))
                            y = model(x)
                        ann, cap, crp, nop, x_seq_rt = pred(input_euler, input_gyro, input_time, x_seq_rt, timestamp_rt, eul_rt, gyro_rt, rt_counter, sess, model, DELAY_SIZE, y, x, midVel, avgVel)
                        print('\nInput [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(input_euler[0],input_euler[1],input_euler[2]))
                        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann[0][0],ann[0][1],ann[0][2]))
                        self.ori_rt = np.concatenate((self.ori_rt, np.array(input_euler).reshape(1,-1)), axis =0)
                        self.ann_pred_rt = np.concatenate((self.ann_pred_rt, ann), axis =0)
                        self.crp_pred_rt = np.concatenate((self.crp_pred_rt, crp), axis =0)
                        self.cap_pred_rt = np.concatenate((self.cap_pred_rt, cap), axis =0)
                        self.nop_pred_rt = np.concatenate((self.nop_pred_rt, nop), axis =0)
                        rt_counter += 1
                    train_counter += 1
                    #visualization of realtime
    #                    if (rt_counter >= 1038):
    #                        y_plot[0] = ori_rt[rt_counter][2]
    #                        y_plot[1] = crp_pred_rt[pastTime][2]
    #                        y_plot[2] = cap_pred_rt[pastTime][2]
    #                        y_plot[3] = nop_pred_rt[pastTime][2]
    #                        y_plot[4] = ann_pred_rt[pastTime][2]
    #                        rtp.RealTimePlot(float(rt_counter/60), y_plot)
    
                    predicted_data = PredictedData(motion_data.timestamp,
                                                   prediction_time,
                                                   predicted_orientation,
                                                   predicted_projection)
                    
                    self.end_prediction(motion_data.timestamp)
                
                    motion_predicted_output.send(predicted_data.pack())
    
                    if self.prediction_output is not None:
                        self.prediction_output.write(motion_data, predicted_data)
                    
                if feedback_recv in dict(events):
                    feedback = await feedback_recv.recv()
                    self.merge_feedback(cbor2.loads(feedback))

    def train_nn(self, eul_train, gyro_train, time_stamp):
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
            
        self.flag = 1
        
        self.midVel = np.nanpercentile(np.abs(velocity_o),50, axis = 0)
        self.avgVel = np.nanmean(np.abs(velocity_o), axis=0)
    #        self.sett(flag, model, DELAY_SIZE)
    
    # process feedbacks
    def start_prediction(self, session):
        assert(session not in self.feedbacks)
        self.feedbacks[session] = {
            'srcmask': 0,
            'startPrediction': time.clock()
        }

    def end_prediction(self, session):
        self.feedbacks[session]['stopPrediction'] = time.clock()

    def merge_feedback(self, feedback):
        if not all(key in feedback for key in ('session', 'source')):
            return

        if not feedback['session'] in self.feedbacks:
            return
        
        session = feedback['session']
        entry = self.feedbacks[session]
        
        if feedback['source'] == 'acli':
            entry['srcmask'] |= 0b01
        elif feedback['source'] == 'asrv':
            entry['srcmask'] |= 0b10
        else:
            return

        del feedback['source']
        self.feedbacks[session] = {**entry, **feedback}

        if entry['srcmask'] == 0b11:
            if self.metric_writer is not None:
                self.metric_writer.write_metric(self.feedbacks[session])

            self.module.feedback_received(self.feedbacks[session])
                
            self.feedbacks = {
                s: self.feedbacks[s] for s in self.feedbacks if s > session
            }