"""Utility functions for habitat extensions.

This module contains utility functions that may be missing in newer versions
of habitat-lab but are required for backward compatibility.
"""


def try_cv2_import():
    r"""The PyRobot python3 version which is a dependency of Habitat-PyRobot integration
    relies on ROS running in python2.7. In order to import cv2 in python3 we need to remove
    the python2.7 path from sys.path. To use the Habitat-PyRobot integration the user
    needs to export environment variable ROS_PATH which will look something like:
    /opt/ros/kinetic/lib/python2.7/dist-packages
    """
    import os
    import sys

    ros_path = os.environ.get("ROS_PATH")
    if ros_path is not None and ros_path in sys.path:
        sys.path.remove(ros_path)
        import cv2

        sys.path.append(ros_path)
    else:
        import cv2

    return cv2
