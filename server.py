import sys
import getopt
from predict_server import PredictModule, MotionPredictServer
from predict_server.simulator import MotionPredictSimulator
import numpy as np
import math
import quaternion
import tensorflow as tf
from realPred_train import pred
from train import train_nn
# from realPred import pred

class App(PredictModule):
    def parse_command_args(self):
        port = feedback = input_file = output = metric_output = None
        
        try:
            opts, args = getopt.getopt(sys.argv[1:], "p:f:m:o:i:")
        except getopt.GetoptError as err:
            print(err)
            sys.exit(1)

        for opt, arg in opts:
            if opt == "-p":
                port = int(arg)
            elif opt == "-f":
                feedback = int(arg)
            elif opt == "-i":
                input_file = arg
            elif opt == "-o":
                output = arg
            elif opt == "-m":
                metric_output = arg
            else:
                assert False, "unhandled option"
                
        return port, feedback, input_file, output, metric_output

    def run(self):
		
        self.rt_counter = 0
        self.x_seq_rt = np.zeros(shape=(6, 9))
        self.timestamp_rt = np.zeros(shape=(2, 1), dtype= float)
        self.gyro_rt = np.zeros(shape=(2, 3), dtype= float)
        self.eul_rt = np.zeros(shape=(2, 3), dtype= float)

        self.flag = 0
        self.trainTime = 2*3600 #number of minutes
        self.train_counter = 0
        self.eul_train = np.zeros((0,3), dtype= float)
        self.gyro_train = np.zeros((0,3), dtype= float)
        self.time_stamp = np.zeros((0,1), dtype= float)
        self.input_nm = 9
#		
        port_input, port_feedback, input_file, output, metric_output = \
            self.parse_command_args()
        if input_file is None:
            assert(port_input is not None and port_feedback is not None)

            server = MotionPredictServer(
                self, port_input, port_feedback, output, metric_output
            )

            try:
                server.run()
                
            except KeyboardInterrupt:
                pass
            finally:                
                server.shutdown()
                
        else:
            assert(output is not None)

            simulator = MotionPredictSimulator(self, input_file, output)

            try:                
                simulator.run()
            except KeyboardInterrupt:
                pass
        
    def quat_to_euler(self, quat):        
        siny_cosp = 2 * (quat[3] * quat[1] - quat[2] * quat[0])
        cosy_cosp = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2])
        yaw = math.atan2(siny_cosp, cosy_cosp)

        sinp = 2 * (quat[3] * quat[2] + quat[0] * quat[1])
        if abs(sinp) >= 1:
            roll = math.copysign(math.pi / 2, sinp)
        else:
            roll = math.asin(sinp)

        sinx_cosp = 2 * (quat[3] * quat[0] - quat[1] * quat[2])
        cosx_cosp = 1 - 2 * (quat[2] * quat[2] + quat[0] * quat[0])
        pitch = math.atan2(sinx_cosp, cosx_cosp)

        return [yaw, pitch, roll]

    def euler_to_quat (self, eul):
        # Convert Euler [oculus] to Quaternion [oculus]
        eul = np.deg2rad(eul)
        X_eul = eul[:,0]
        Y_eul = eul[:,1]
        Z_eul = eul[:,2]

        cos_pitch, sin_pitch = np.cos(X_eul/2), np.sin(X_eul/2)
        cos_yaw, sin_yaw = np.cos(Y_eul/2), np.sin(Y_eul/2)
        cos_roll, sin_roll = np.cos(Z_eul/2), np.sin(Z_eul/2)

        # order: w,x,y,z
        # quat = unit_q(quat)
        quat = np.nan * np.ones( (eul.shape[0],4) )
        quat[:,0] = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
        quat[:,1] = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
        quat[:,2] = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll 
        quat[:,3] = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll     
        return (quat[:,[3,1,2,0]])
        
    def Robust_overfilling(self, input_orientation, prediction, input_projection, offset = 1, fixed_param = 1):
        #Get Input Projection Distance for each side in x,y coordinate
        IPDx = input_projection[2]-input_projection[0]
        IPDy = input_projection[1]-input_projection[3]
    	
    	#Define projection distance to the user
        h = 1
    	
    	#Get the corner point distance to the rotation center for each side in x,y coordinate
        rx = np.sqrt(h**2+1/4*IPDx**2)
        ry = np.sqrt(h**2+1/4*IPDy**2)
    	
    	#Get initial input angle to the rotational center
        input_anglex = np.arctan(IPDx/(2*h))
        input_angley = np.arctan(IPDx/(2*h))
    	
    	#Get user's direction based on prediction motion
        pitch_diff = (prediction[0]-input_orientation[0])*np.pi/180
        roll_diff = (prediction[1]-input_orientation[1])*np.pi/180
        yaw_diff = (prediction[2]-input_orientation[2])*np.pi/180
    	
    	#Calculate predicted margin based on translation movement
        x_r = max(input_projection[2],rx*abs(np.sin(input_anglex-yaw_diff)))
        x_l = min(input_projection[0],-rx*abs(np.sin(input_anglex+yaw_diff)))
        y_t = max(input_projection[1],ry*abs(np.sin(input_angley+pitch_diff)))
        y_b = min(input_projection[3],-ry*abs(np.sin(input_angley-pitch_diff)))
    
    	#Calculate predicted margin based on rotational movement
        x_rr = rx*abs(np.sin(input_anglex+abs(roll_diff)))
        x_ll = -rx*abs(np.sin(input_anglex+abs(roll_diff)))
        y_tt = ry*abs(np.sin(input_angley+abs(roll_diff)))
        y_bb = -ry*abs(np.sin(input_angley+abs(roll_diff)))
    	
    	#Calculate final movement
        p_r = x_r+x_rr-IPDx/2
        p_l = x_l+x_ll+IPDx/2
        p_t = y_t+y_tt-IPDy/2
        p_b = y_b+y_bb+IPDy/2
    
        'Enhancement'
    ##	#get largest point
    #	z_yaw = abs(h-rx*np.cos(input_anglex+abs(yaw_diff)))
    #	z_pitch = abs(h-ry*np.cos(input_angley+abs(pitch_diff)))
    #	
    #	if (yaw_diff<0):
    #		p_r = np.sqrt(abs(p_r-p_l)**2+z_yaw**2)+p_l
    #	else: 
    #		p_l = p_r-np.sqrt(abs(p_l-p_r)**2+z_yaw**2)
    #	if (pitch_diff>0):
    #		p_t = np.sqrt(abs(p_t-p_b)**2+z_pitch**2)+p_b
    #	else :
    #		p_b = p_t-np.sqrt(abs(p_b-p_t)**2+z_pitch**2)
    	
    	#Calculate margin based on genrated area
        margin = np.sqrt((p_r-p_l)*(p_t-p_b)*offset)/2
        p_l = -(margin+np.sin(yaw_diff))
        p_t = margin+np.sin(pitch_diff)
        p_r = margin-np.sin(yaw_diff)
        p_b = -(margin-np.sin(pitch_diff))
    	
        'Enhancement ver 2'
    ##	Get dilation on high velocity
        p_r = max(p_r*(np.sin(abs(yaw_diff))*(fixed_param-1)+1),p_r*(np.sin(abs(pitch_diff))*(fixed_param-1)+1))
        p_l = min(p_l*(np.sin(abs(yaw_diff))*(fixed_param-1)+1),p_l*(np.sin(abs(pitch_diff))*(fixed_param-1)+1))
        p_t = max(p_t*(np.sin(abs(pitch_diff))*(fixed_param-1)+1),p_t*(np.sin(abs(yaw_diff))*(fixed_param-1)+1))
        p_b = min(p_b*(np.sin(abs(pitch_diff))*(fixed_param-1)+1), p_b*(np.sin(abs(yaw_diff))*(fixed_param-1)+1))
    	
    
    	#Shifting to predicted point
    #	return [-(abs(p_l)+np.sin(yaw_diff)),abs(p_t)+np.sin(pitch_diff),abs(p_r)-np.sin(yaw_diff),-(abs(p_b)-np.sin(pitch_diff))]
    
        return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
        # margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
        # return [-margins, margins, margins, -margins]
    
    def calc_optimal_overhead(self, hmd_orientation, frame_orientation, hmd_projection):
        q_d = np.matmul(
                np.linalg.inv(quaternion.as_rotation_matrix(hmd_orientation)),
                quaternion.as_rotation_matrix(frame_orientation)
                )
        #Projection Orientation:
            #hmd_projection[0] : Left (Negative X axis)
            #hmd_projection[1] : Top (Positive Y axis)
            #hmd_projection[2] : Right (Positive X axis)
            #hmd_projection[3] : Bottom (Negative Y axis)
            
        lt = np.matmul(q_d, [hmd_projection[0], hmd_projection[1], 1])
        p_lt = np.dot(lt, 1 / lt[2])
        
        rt = np.matmul(q_d, [hmd_projection[2], hmd_projection[1], 1])
        p_rt = np.dot(rt, 1 / rt[2])
        
        rb = np.matmul(q_d, [hmd_projection[2], hmd_projection[3], 1])
        p_rb = np.dot(rb, 1 / rb[2])
        
        lb = np.matmul(q_d, [hmd_projection[0], hmd_projection[3], 1])
        p_lb = np.dot(lb, 1 / lb[2])
        
        p_l = min(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
        p_t = max(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
        p_r = max(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
        p_b = min(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
        
    # 	size = max(p_r - p_l, p_t - p_b)
    # 	a_overfilling = size * size
        
    # 	a_hmd = (hmd_projection[2] - hmd_projection[0]) * (hmd_projection[1] - hmd_projection[3])
        
        return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
	
	# implements PredictModule
    '''Prediction with training'''
    def predict(self, motion_data, sess):
        # no prediction
        prediction_time = 300.0  # ms
		
        input_euler = self.quat_to_euler(motion_data.orientation)#self.get_euler()
        input_euler = np.array([input_euler[1],input_euler[2],input_euler[0]])*180/np.pi
        self.input_gyro = np.array(motion_data.angular_velocities)*180/np.pi
        input_time = motion_data.timestamp/ 705600000
        
        if (self.flag == 0):
            predicted_orientation = motion_data.orientation
            self.overfilling = motion_data.camera_projection
            ann = cap = crp = np.array(input_euler).reshape(1,-1)
        
        if (self.flag == 0 and self.train_counter >= 1500):
            self.eul_train = np.concatenate((self.eul_train, np.array(input_euler).reshape(1,-1)), axis =0)
            self.gyro_train = np.concatenate((self.gyro_train, np.array(self.input_gyro).reshape(1,-1)), axis =0)
            self.time_stamp = np.concatenate((self.time_stamp, np.array(input_time).reshape(1,-1)), axis =0)            
            print(self.train_counter)
            if (self.train_counter == 1500+self.trainTime):
                self.flag, self.model, self.DELAY_SIZE, self.midVel, self.avgVel = train_nn(self.eul_train, self.gyro_train, self.time_stamp)#trainInit.get()
#                print(self.midVel)
#                print(self.avgVel)
                self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.DELAY_SIZE, self.input_nm))
                self.y = self.model(self.x)
            
        elif (self.flag == 1):# get the data after 17 seconds (60 FPS is assumed)
            # Overhead Orientation
            idx = [0,3,1,2]
            
            ann, cap, crp, nop, self.x_seq_rt = pred(input_euler, self.input_gyro, input_time, self.x_seq_rt, self.timestamp_rt, self.eul_rt, self.gyro_rt, self.rt_counter, sess, self.model, self.DELAY_SIZE, self.y, self.x, self.midVel, self.avgVel)

            quat_quat_predict = self.euler_to_quat(ann)
            quat_quat_data = self.euler_to_quat(input_euler.reshape(1,-1))
            projection_input = motion_data.camera_projection
            
            # hmd_orientation = quat_quat_data[:,idx]
            # frame_orientation = quat_quat_predict[:,idx]
            
            # #Calculate overfiling
            # input_orientation = np.quaternion(
            #     hmd_orientation[:,0],
            #     hmd_orientation[:,1],
            #     hmd_orientation[:,2],
            #     hmd_orientation[:,3]
            #     )
            # predict_orientation = np.quaternion(
            #     frame_orientation[:,0],
            #     frame_orientation[:,1],
            #     frame_orientation[:,2],
            #     frame_orientation[:,3]
            #     )
            input_projection = projection_input

            # margin = self.calc_optimal_overhead(input_orientation, predict_orientation, input_projection)
            margin = self.Robust_overfilling(input_euler, ann[0], input_projection, offset = 1.1, fixed_param = 1.3)
#            margin[0] = margin[0]+motion_data.camera_projection[0]
#            margin[1] = margin[1]-motion_data.camera_projection[1]
#            margin[2] = margin[2]-motion_data.camera_projection[2]
#            margin[3] = margin[3]+motion_data.camera_projection[3]
            
            print('\nInput [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(input_euler[0],input_euler[1],input_euler[2]))
            print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann[0][0],ann[0][1],ann[0][2]))
            print('Overfilling: {:.2f},{:.2f},{:.2f},{:.2f}'.format(margin[0],margin[1], margin[2], margin[3]))
            self.overfilling = [margin[0], margin[1], margin[2], margin[3]]
            
            self.rt_counter += 1
            
            predicted_orientation = quat_quat_predict[0]
            # predicted_orientation = quat_quat_data[0]
            # predicted_orientation = motion_data.orientation
            
            # print(quat_quat_predict[0])
            # print(quat_quat_data[0])
            # print(motion_data.orientation)
            
        self.train_counter += 1
        # overfilling delta in radian (left, top, right, bottom)
        projection_output = self.overfilling

        return prediction_time, predicted_orientation, projection_output, input_euler, ann, cap, crp, self.flag

    '''Prediction without training'''
#    def predict(self, motion_data, sess):
#         # no prediction
        
#         prediction_time = 300.0  # ms
# 		
#         input_euler = self.quat_to_euler(motion_data.orientation)#self.get_euler()
#         input_euler = np.array([input_euler[1],input_euler[2],input_euler[0]])*180/np.pi
#         input_gyro = np.array(motion_data.angular_velocities)*180/np.pi
#         input_time = motion_data.timestamp/ 705600000
        
#         if (self.flag == 0):
#             predicted_orientation = motion_data.orientation
#             self.overfilling = motion_data.camera_projection
#             ann = cap = crp = np.array(input_euler).reshape(1,-1)

#         if (self.rt_counter >= 1500):
#             self.flag = 1
#             idx = [0,3,1,2]
            
#             ann, cap, crp, nop, self.x_seq_rt = pred(input_euler, input_gyro, input_time, self.x_seq_rt, self.timestamp_rt,  self.gyro_rt, self.rt_counter, sess)
#             quat_quat_predict = self.euler_to_quat(ann)
#             quat_quat_data = self.euler_to_quat(input_euler.reshape(1,-1))
#             projection_input = motion_data.camera_projection
            
#             hmd_orientation = quat_quat_data[:,idx]
#             frame_orientation = quat_quat_predict[:,idx]
            
#             #Calculate overfiling
#             input_orientation = np.quaternion(
#                 hmd_orientation[:,0],
#                 hmd_orientation[:,1],
#                 hmd_orientation[:,2],
#                 hmd_orientation[:,3]
#                 )
#             predict_orientation = np.quaternion(
#                 frame_orientation[:,0],
#                 frame_orientation[:,1],
#                 frame_orientation[:,2],
#                 frame_orientation[:,3]
#                 )
#             input_projection = projection_input
    
#             margin = self.calc_optimal_overhead(input_orientation, predict_orientation, input_projection)
#             # margin = self.Robust_overfilling(input_euler, ann, input_projection, offset = 1, fixed_param = 1)
#     #            margin[0] = margin[0]+motion_data.camera_projection[0]
#     #            margin[1] = margin[1]-motion_data.camera_projection[1]
#     #            margin[2] = margin[2]-motion_data.camera_projection[2]
#     #            margin[3] = margin[3]+motion_data.camera_projection[3]
            
#             print('\nInput [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(input_euler[0],input_euler[1],input_euler[2]))
#             print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann[0][0],ann[0][1],ann[0][2]))
#             print('Overfilling: {:.2f},{:.2f},{:.2f},{:.2f}'.format(margin[0],margin[1], margin[2], margin[3]))
#             self.overfilling = [margin[0], margin[1], margin[2], margin[3]]
            
#             predicted_orientation = quat_quat_predict[0]
#             # predicted_orientation = quat_quat_data[0]
#             # predicted_orientation = motion_data.orientation
        
#         self.rt_counter += 1
        
#         # print(quat_quat_predict[0])
#         # print(quat_quat_data[0])

#         # overfilling delta in radian (left, top, right, bottom)
#         projection_output = self.overfilling

#         return prediction_time, predicted_orientation, projection_output, input_euler, ann, cap, crp, self.flag

    def feedback_received(self, feedback):
        # see PrefMetricWriter.write_metric() to understand feedback values
        # (motion_prediction_server.py:320)
        
        # example : calculate overall latency
        #
        # overall_latency = feedback['endClientRender'] - feedback['gatherInput']
        
        pass
    
    
def main():
    app = App()
    app.run()

	
def ambil_euler(MotionPredictServer):
	nilai_euler = MotionPredictServer.get_euler
	return nilai_euler

if __name__ == "__main__":
    main()
