#!/usr/bin/env python3
# coding=utf-8

import argparse
import time

import rclpy
from rclpy.node import Node
import tf_transformations  # You may need to install this package
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np

class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('publish_initial_pose')
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)

    def publish_initial_pose(self, x, y, z, q1, q2, q3, q4):
        quat = [q1, q2, q3, q4]
        xyz = [x, y, z]
        msg = PoseWithCovarianceStamped()
        # msg.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        msg.pose.pose.position.x = xyz[0]
        msg.pose.pose.position.y = xyz[1]
        msg.pose.pose.position.z = xyz[2]
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        self.pub_pose.publish(msg)
        self.get_logger().info('Initial Pose: {} {} {} {} {} {} {}'.format(x, y, z, q1, q2, q3, q4))

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parser.add_argument('z', type=float)
    parser.add_argument('q1', type=float)
    parser.add_argument('q2', type=float)
    parser.add_argument('q3', type=float)
    parser.add_argument('q4', type=float)
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = InitialPosePublisher()
    # Wait a moment for publishers to connect
    time.sleep(1)
    node.publish_initial_pose(parsed_args.x, parsed_args.y, parsed_args.z,
                              parsed_args.q1, parsed_args.q2, parsed_args.q3, parsed_args.q4)
    # Allow time for the message to be sent before shutting down.
    time.sleep(1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()