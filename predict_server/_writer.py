import math
import numpy as np
import quaternion

from abc import abstractmethod, ABCMeta


class CsvWriter(metaclass=ABCMeta):
    def __init__(self, output):
        self.output = open(output, 'w')
        self.write_line(self.make_header_items())

    def write_line(self, items):
        self.output.write(','.join(items) + '\n')

    def close(self):
        self.output.close()

    @abstractmethod
    def make_header_items(self):
        pass

    
class PredictionOutputWriter(CsvWriter):
    def __init__(self, output):
        super().__init__(output)

    def make_header_items(self):
        return [
            'timestamp',
            'biosignal_0',
            'biosignal_1',
            'biosignal_2',
            'biosignal_3',
            'biosignal_4',
            'biosignal_5',
            'biosignal_6',
            'biosignal_7',
            'acceleration_x',
            'acceleration_y',
            'acceleration_z',
            'angular_vec_x',
            'angular_vec_y',
            'angular_vec_z',
            'magnetic_x',
            'magnetic_y',
            'magnetic_z',
            'input_orientation_x',
            'input_orientation_y',
            'input_orientation_z',
            'input_orientation_w',
            'input_orientation_yaw',
            'input_orientation_pitch',
            'input_orientation_roll',
            'input_projection_left',
            'input_projection_top',
            'input_projection_right',
            'input_projection_bottom',
            'prediction_time',
            'predicted_orientation_x',
            'predicted_orientation_y',
            'predicted_orientation_z',
            'predicted_orientation_w',
            'predicted_orientation_yaw',
            'predicted_orientation_pitch',
            'predicted_orientation_roll',
            'predicted_projection_left',
            'predicted_projection_top',
            'predicted_projection_right',
            'predicted_projection_bottom'
        ]

    def write(self, motion_data, predicted_data):
        input_orientation_euler = self.quat_to_euler(motion_data.orientation)
        predicted_orientation_euler = self.quat_to_euler(predicted_data.orientation)
        
        self.write_line([
            str(motion_data.timestamp),
            str(motion_data.biosignal[0]),
            str(motion_data.biosignal[1]),
            str(motion_data.biosignal[2]),
            str(motion_data.biosignal[3]),
            str(motion_data.biosignal[4]),
            str(motion_data.biosignal[5]),
            str(motion_data.biosignal[6]),
            str(motion_data.biosignal[7]),
            str(motion_data.acceleration[0]),
            str(motion_data.acceleration[1]),
            str(motion_data.acceleration[2]),
            str(motion_data.angular_velocities[0]),
            str(motion_data.angular_velocities[1]),
            str(motion_data.angular_velocities[2]),
            str(motion_data.magnetic_field[0]),
            str(motion_data.magnetic_field[1]),
            str(motion_data.magnetic_field[2]),
            str(motion_data.orientation[0]),
            str(motion_data.orientation[1]),
            str(motion_data.orientation[2]),
            str(motion_data.orientation[3]),
            str(input_orientation_euler[0]),
            str(input_orientation_euler[1]),
            str(input_orientation_euler[2]),
            str(motion_data.camera_projection[0]),
            str(motion_data.camera_projection[1]),
            str(motion_data.camera_projection[2]),
            str(motion_data.camera_projection[3]),
            str(predicted_data.prediction_time),
            str(predicted_data.orientation[0]),
            str(predicted_data.orientation[1]),
            str(predicted_data.orientation[2]),
            str(predicted_data.orientation[3]),
            str(predicted_orientation_euler[0]),
            str(predicted_orientation_euler[1]),
            str(predicted_orientation_euler[2]),
            str(predicted_data.camera_projection[0]),
            str(predicted_data.camera_projection[1]),
            str(predicted_data.camera_projection[2]),
            str(predicted_data.camera_projection[3])
        ])

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

        
class PerfMetricWriter(CsvWriter):
    def __init__(self, output):
        super().__init__(output)

    def make_header_items(self):
        return [
            'timestamp',
            'input_orientation_x',
            'input_orientation_y',
            'input_orientation_z',
            'input_orientation_w',
            'input_projection_left',
            'input_projection_top',
            'input_projection_right',
            'input_projection_bottom',
            'predicted_orientation_x',
            'predicted_orientation_y',
            'predicted_orientation_z',
            'predicted_orientation_w',
            'predicted_projection_left',
            'predicted_projection_top',
            'predicted_projection_right',
            'predicted_projection_bottom',
            'overall_latency',
            'gather_input_start_prediction',
            'start_prediction_send_predicted',
            'send_predicted_start_server_render',
            'start_server_render_start_encode',
            'start_encode_send_video',
            'send_video_start_recv_video',
            'start_recv_video_start_decode',
            'start_decode_start_client_render',
            'start_client_render_end_client_render',
            'frame_type',
            'frame_size',
            'optimal_overhead',
            'actual_overhead'
        ]

    def write_metric(self, feedback):
        # latency
        overall_latency = feedback['endClientRender'] - feedback['gatherInput']
        
        start_prediction_send_predicted = \
            feedback['stopPrediction'] - feedback['startPrediction']
        send_predicted_start_server_render = \
            feedback['startServerRender'] - feedback['startSimulation']
        start_server_render_start_encode = \
            feedback['startEncode'] - feedback['startServerRender']
        start_encode_send_video = \
            feedback['sendVideo'] - feedback['startEncode']
        start_recv_video_start_decode = \
            feedback['startDecode'] - feedback['firstFrameReceived']
        start_decode_start_client_render = \
            feedback['startClientRender'] - feedback['startDecode']
        start_client_render_end_client_render = \
            feedback['endClientRender'] - feedback['startClientRender']
        
        rtt = overall_latency - (
            start_prediction_send_predicted +
            send_predicted_start_server_render +
            start_server_render_start_encode +
            start_encode_send_video +
            start_recv_video_start_decode +
            start_decode_start_client_render +
            start_client_render_end_client_render
        )

        gather_input_start_prediction = send_video_start_recv_video = rtt / 2

        # overhead
        hmd_orientation = np.quaternion(
            feedback['hmdOrientationW'],
            feedback['hmdOrientationX'],
            feedback['hmdOrientationY'],
            feedback['hmdOrientationZ'],
        )
        hmd_projection = [
            feedback['hmdProjectionL'],
            feedback['hmdProjectionT'],
            feedback['hmdProjectionR'],
            feedback['hmdProjectionB']
        ]
        frame_orientation = np.quaternion(
            feedback['frameOrientationW'],
            feedback['frameOrientationX'],
            feedback['frameOrientationY'],
            feedback['frameOrientationZ']
        )
        frame_projection = [
            feedback['frameProjectionL'],
            feedback['frameProjectionT'],
            feedback['frameProjectionR'],
            feedback['frameProjectionB']
        ]

        self.write_line([
            str(feedback['session']),
            str(feedback['hmdOrientationX']),
            str(feedback['hmdOrientationY']),
            str(feedback['hmdOrientationZ']),
            str(feedback['hmdOrientationW']),
            str(feedback['hmdProjectionL']),
            str(feedback['hmdProjectionT']),
            str(feedback['hmdProjectionR']),
            str(feedback['hmdProjectionB']),
            str(feedback['frameOrientationX']),
            str(feedback['frameOrientationY']),
            str(feedback['frameOrientationZ']),
            str(feedback['frameOrientationW']),
            str(feedback['frameProjectionL']),
            str(feedback['frameProjectionT']),
            str(feedback['frameProjectionR']),
            str(feedback['frameProjectionB']),
            str(overall_latency),
            str(gather_input_start_prediction),
            str(start_prediction_send_predicted),
            str(send_predicted_start_server_render),
            str(start_server_render_start_encode),
            str(start_encode_send_video),
            str(send_video_start_recv_video),
            str(start_recv_video_start_decode),
            str(start_decode_start_client_render),
            str(start_client_render_end_client_render),
            str("{:.0f}".format(feedback['frameType'])),
            str("{:.0f}".format(feedback['frameSize'])),
            str(self.calc_optimal_overhead(hmd_orientation, frame_orientation, hmd_projection)),
            str(self.calc_actual_overhead(hmd_projection, frame_projection))
        ])
		

    def calc_optimal_overhead(self, hmd_orientation, frame_orientation, hmd_projection):
        q_d = np.matmul(
            np.linalg.inv(quaternion.as_rotation_matrix(hmd_orientation)),
            quaternion.as_rotation_matrix(frame_orientation)
        )

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

        size = max(p_r - p_l, p_t - p_b)
        a_overfilling = size * size

        a_hmd = (hmd_projection[2] - hmd_projection[0]) * (hmd_projection[1] - hmd_projection[3])

        return a_overfilling / a_hmd - 1

    def calc_actual_overhead(self, hmd_projection, frame_projection):
        a_hmd = (hmd_projection[2] - hmd_projection[0]) * (hmd_projection[1] - hmd_projection[3])
        a_overfilling = (frame_projection[2] - frame_projection[0]) * (frame_projection[1] - frame_projection[3])

        return a_overfilling / a_hmd - 1
