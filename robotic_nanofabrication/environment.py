import socket
import struct
import numpy as np
import logging
import copy

import util

logger = logging.getLogger('general')

# Logger just for received measurements
measurement_logger = logging.getLogger("measurement")


# This is the environment used when connecting to the real
# scanning tunneling microscope (i. e. not a simulation)
class STMEnvironment:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.position = np.array([[0., 0., 0.]], dtype=np.float32)

        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.host, self.port)

        logger.info('connect to %s port %s' % server_address)
        self.connection.connect(server_address)

        logger.info('connected')

    def __exit__(self, type, value, traceback):
        self.connection.close()

    def _recvall(self, count):
        buf = b''
        while count:
            newbuf = self.connection.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def recvfloat(self):
        buf = self._recvall(4)
        return struct.unpack('>f', buf)

    def sendfloat(self, value):
        buf = struct.pack('>f', value)
        self.connection.sendall(buf)

    def send_action(self, step):
        # the coordinate system of the microscope is mirrored in z,
        # and we only adjust it here
        step = copy.copy(step)
        step[2] *= -1
        for x in step:
            self.sendfloat(float(x))
        logger.info('ACTION: ' + str(step))

    def receive_measurement(self):
        measurement = None
        received_signal = self.recvfloat()
        logger.info('received signal "%s"' % received_signal)

        if received_signal[0] == 1000001.0:
            logger.info("Success!")
            measurement =  "success"
        elif received_signal[0] == 1000002.0:
            logger.info("Failure")
            measurement =  "fail"
        elif received_signal[0] == 1000003.0:
            logger.info("Received signal to delete last episode")
            measurement =  "delete_last_episode"
        else:
            current = received_signal
            df = self.recvfloat()
            logger.info('received current "%s"' % current)
            logger.info('received df "%s"' % df)
            measurement = current, df

        measurement_logger.info(measurement)
        return measurement

    def reset(self):
        pass


# This environment was used in early simulation experiments. Later we created
# a more sophisticated simulation (see below), based on more realistic assumptions
class ToyTrajectoryEnvironment:
    def __init__(self, trajectory_path, lower_hose_radius=0.5, project_angle=None):
        ''' Load trajectory'''
        self.lower_hose_radius = lower_hose_radius
        self.success_height = 14

        with open(trajectory_path, "r") as f:
            self.trajectory = np.loadtxt(f)
        self.trajectory -= self.trajectory[0]
        self.trajectory[:, 2] *= -1
        if project_angle is not None:
            self.trajectory[:, :2] = util.project_into_plane(self.trajectory[:, :2],
                                                             angle=project_angle)
        self.position = np.array([0., 0., 0.])

    def send_action(self, step):
        # update position
        self.position += step

    def reset(self):
        self.position = np.array([0., 0., 0.])

    # the _check_fail method can be overridden by other environments
    # to create different simulations
    def _check_fail(self):
        # find elements in trajectry that are closest to current position
        dist = np.linalg.norm(self.position - self.trajectory, axis=1)
        closest_distance = np.min(dist)

        if closest_distance > self.lower_hose_radius:
            return True
        else:
            return False

    def receive_measurement(self):
        measurement = None
        if self.position[2] >= self.success_height:
            measurement =  "success"
        # to test proper handling of deleted episodes
        elif np.random.choice([True, False], p=[0.0, 1.0]):
            measurement =  "delete_last_episode"
        else:
            if self._check_fail():
                measurement =  "fail"
            else:
                measurement =  0., 0.

        measurement_logger.info(measurement)
        return measurement


class SophisticatedToyTrajectoryEnvironment(ToyTrajectoryEnvironment):
    def __init__(self, trajectory_path, lower_hose_radius=0.5,
                 upper_hose_radius=2.0, project_angle=None):
        ''' Load trajectory'''
        self.lower_hose_radius = lower_hose_radius
        self.upper_hose_radius = upper_hose_radius
        self.success_height = 14

        with open(trajectory_path, "r") as f:
            self.trajectory = np.loadtxt(f)
        self.trajectory -= self.trajectory[0]
        self.trajectory[:, 2] *= -1
        if project_angle is not None:
            self.trajectory[:, :2] = util.project_into_plane(self.trajectory[:, :2],
                                                             angle=project_angle)
        self.position = np.array([0., 0., 0.])

    def _check_fail(self):
        lower_ball = 3  # lower ball of acceptance
        upper_ball = 13.5  # upper ball of acceptance
        lower_hose_length = 2
        upper_hose_length = 2

        r = np.linalg.norm(self.position)  # distance to origin
        # find element in trajectry that is closest to current position
        dist = np.linalg.norm(self.position - self.trajectory, axis=1)
        x = np.min(dist)  # distance to trajectory

        # diagonal plane through the lower ball defined by n * x - b = 0
        lower_ball_pos = np.copy(self.trajectory[np.argmin(np.abs(np.linalg.norm(self.trajectory, axis=1) - lower_ball))])
        lower_ball_pos[2] += 1.5 * self.lower_hose_radius
        opposite_pos = np.copy(lower_ball_pos)
        opposite_pos[2] = 0
        opposite_pos = - opposite_pos / np.linalg.norm(opposite_pos[:2]) * lower_ball
        # find the point on the line closest to the origin
        # https://math.stackexchange.com/questions/2193720/find-a-point-on-a-line-segment-which-is-the-closest-to-other-point-not-on-the-li#2193726
        A = lower_ball_pos
        B = opposite_pos
        v = B - A
        u = A  # should subtract P but P is (0,0,0)
        t = - (v @ u) / (v @ v)
        P = (1 - t) * A + t * B
        b = np.linalg.norm(P)
        n = P / b

        if ((r <= lower_ball) and (n @ self.position - b <= 0)) or \
           ((r >= lower_ball) and (r <= lower_ball + lower_hose_length) and (x <= self.lower_hose_radius)) or \
           ((r >= lower_ball) and (r <= upper_ball) and (x <= self.lower_hose_radius + r - lower_ball - lower_hose_length)) or \
           ((r >= upper_ball) and (r <= upper_ball + upper_hose_length) and (x <= self.upper_hose_radius)) or \
           ((r >= upper_ball + upper_hose_length) and (x <= self.upper_hose_radius + (r - upper_ball - upper_hose_length))):
            return False  # no rupture
        else:
            return True  # rupture
