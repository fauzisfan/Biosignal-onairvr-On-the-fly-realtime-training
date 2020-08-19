import sys
import getopt
from predict_server import PredictModule, MotionPredictServer
from predict_server.simulator import MotionPredictSimulator

from predict_server import __init__

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
        
    # implements PredictModule
    def predict(self, motion_data):
        # no prediction
        prediction_time = 100.0  # ms
        predicted_orientation = motion_data.orientation

        # overfilling delta in radian (left, top, right, bottom)
        overfilling = [0.0, 0.0, 0.0, 0.0]

        return prediction_time, predicted_orientation, self.make_camera_projection(motion_data, overfilling)

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
