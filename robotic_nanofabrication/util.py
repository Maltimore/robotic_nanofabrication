import numpy as np
import signal
import logging
import subprocess
import warnings

logger = logging.getLogger('general')


def radians(angle):
    return angle / 360 *  2 * np.pi


def project_into_plane(arr, angle):
    """project_into_plane

    :param arr: array-like, shape (..., 2), or (2,)
        contains samples in rows and x, y position in columns
    :param angle: scalar, angle with x-axis in which to project (radians)

    :Returns: array, shape like arr (input array)
    """
    orig_shape = arr.shape
    arr = arr.reshape((-1, 2))

    projection_vec = np.array([np.cos(angle), np.sin(angle)]).reshape((2, 1))
    projection = np.dot(arr, projection_vec) * projection_vec.reshape((1, 2))
    return projection.reshape(orig_shape)


class Ignore_KeyboardInterrupt:
    def __enter__(self):
        def handle_exception(signal_, frame):
            logger.warning('')
            logger.warning('You tried to interrupt the program. Currently this is not a good time.')
            logger.warning('Wait a few seconds, please.')
            logger.warning('Otherwise, interrupt again and live with the consequences.')
            # reinstate the original handler in case we get another interrupt
            signal.signal(signal.SIGINT, self.original_handler)
        self.original_handler = signal.signal(signal.SIGINT, handle_exception)

    def __exit__(self, exc_type, exc_value, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip('\n')
    except Exception:
        warnings.warn('\nWarning! Failed to get git revision hash!\n')
        return('FAILED_to_get_git_revision_hash')
