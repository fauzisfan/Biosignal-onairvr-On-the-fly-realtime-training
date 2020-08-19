import csv
from ._types import MotionData, PredictedData
from ._writer import PredictionOutputWriter


class MotionPredictSimulator:
    def __init__(self, module, input_motion_data, prediction_output):
        self.module = module
        self.input_motion_data = input_motion_data
        self.prediction_output = PredictionOutputWriter(
            prediction_output
        ) if prediction_output is not None else None

    def run(self):
        with open(self.input_motion_data, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                motion_data = MotionData(
                    0,
                    [float(row["biosignal_0"]), float(row["biosignal_1"]),
                     float(row["biosignal_2"]), float(row["biosignal_3"]),
                     float(row["biosignal_4"]), float(row["biosignal_5"]),
                     float(row["biosignal_6"]), float(row["biosignal_7"])],
                    [float(row["acceleration_x"]),
                     float(row["acceleration_y"]),
                     float(row["acceleration_z"])],
                    [float(row["angular_vec_x"]),
                     float(row["angular_vec_y"]),
                     float(row["angular_vec_z"])],
                    [float(row["magnetic_x"]),
                     float(row["magnetic_y"]),
                     float(row["magnetic_z"])],
                    [float(row["input_orientation_x"]),
                     float(row["input_orientation_y"]),
                     float(row["input_orientation_z"]),
                     float(row["input_orientation_w"])],
                    float(row["timestamp"]),
                    [float(row["input_projection_left"]),
                     float(row["input_projection_top"]),
                     float(row["input_projection_right"]),
                     float(row["input_projection_bottom"])]
                )

                prediction_time, predicted_orientation, predicted_projection = \
                    self.module.predict(motion_data)
                
                predicted_data = PredictedData(motion_data.timestamp,
                                               prediction_time,
                                               predicted_orientation,
                                               predicted_projection)

                if self.prediction_output is not None:
                    self.prediction_output.write(motion_data, predicted_data)
