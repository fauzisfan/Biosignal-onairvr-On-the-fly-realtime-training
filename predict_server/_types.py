import math
import struct


class MotionData:
    @classmethod
    def from_bytes(cls, bytes):
        def float_list_from_bytes(bytes, count, pos):
            return list(
                struct.unpack_from('>{}f'.format(count), bytes, pos)
            ), pos + 4 * count
        
        pos = 0

        header, sample_number = struct.unpack_from('BB', bytes, pos)
        assert((header & 0xFF) == 0xA0)
        pos += 2

        biosignal, pos = float_list_from_bytes(bytes, 8, pos)
        acceleration, pos = float_list_from_bytes(bytes, 3, pos)
        angular_velocities, pos = float_list_from_bytes(bytes, 3, pos)
        magnetic_field, pos = float_list_from_bytes(bytes, 3, pos)
        orientation, pos = float_list_from_bytes(bytes, 4, pos)

        timestamp = struct.unpack_from('>q', bytes, pos)[0]
        pos += 8

        footer = struct.unpack_from('B', bytes, pos)[0]
        assert((footer & 0xF8) == 0xC0)
        pos += 1

        camera_projection, pos = float_list_from_bytes(bytes, 4, pos)

        return cls(
            sample_number, biosignal, acceleration, angular_velocities,
            magnetic_field, orientation, timestamp, camera_projection
        )
    
    def __init__(self, sample_number, biosignal, acceleration,
                 angular_velocities, magnetic_field,
                 orientation, timestamp, camera_projection):
        self.sample_number = sample_number
        self.biosignal = biosignal
        self.acceleration = acceleration
        self.angular_velocities = angular_velocities
        self.magnetic_field = magnetic_field
        self.orientation = orientation
        self.timestamp = timestamp
        self.camera_projection = camera_projection

    def fov(self):
        return math.atan(self.camera_projection[1]) + math.atan(-self.camera_projection[3])

    def __str__(self):
        return (
            "sample {0}, biosignal {1}, acceleration {2}, "
            "angular velocities {3}, magnetic field {4}, orientation {5}, "
            "timestamp {6}, fov {7}"
        ).format(
            self.sample_number, self.biosignal, self.acceleration,
            self.angular_velocities, self.magnetic_field, self.orientation,
            self.timestamp, self.fov()
        )

    
class PredictedData:
    def __init__(self, timestamp, prediction_time, orientation, camera_projection):
        self.timestamp = timestamp
        self.prediction_time = prediction_time
        self.orientation = orientation
        self.camera_projection = camera_projection

    def pack(self):
        return struct.pack(
            '>q9f',
            self.timestamp,
            self.prediction_time,
            self.orientation[0],
            self.orientation[1],
            self.orientation[2],
            self.orientation[3],
            self.camera_projection[0],
            self.camera_projection[1],
            self.camera_projection[2],
            self.camera_projection[3]
        )
