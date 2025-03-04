#!/usr/bin/env python3
# coding=utf-8

import copy
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
import tf_transformations as tf_trans  # pip install tf-transformations if needed
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry

# Global variables (you may encapsulate these as members of your Node subclass)
cur_odom_to_baselink = None
cur_map_to_odom = None

def pose_to_mat(pose_msg):
    pose = pose_msg.pose.pose.position
    ori = pose_msg.pose.pose.orientation
    
    trans_mat = np.array([
        [1, 0, 0, pose.x],
        [0, 1, 0, pose.y],
        [0, 0, 1, pose.z],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    rot_mat = tf_trans.quaternion_matrix([ori.x, ori.y, ori.z, ori.w])
    
    return trans_mat @ rot_mat

class TransformFusionNode(Node):
    def __init__(self):
        super().__init__('transform_fusion')
        self.get_logger().info('Transform Fusion Node Inited...')
        # Publishers and subscribers
        self.pub_localization = self.create_publisher(Odometry, '/localization', 1)
        self.create_subscription(Odometry, '/Odometry', self.cb_save_cur_odom, 1)
        self.create_subscription(Odometry, '/map_to_odom', self.cb_save_map_to_odom, 1)
        # Create a tf2 TransformBroadcaster using this node
        self.br = TransformBroadcaster(self)
        # Frequency (Hz)
        self.freq_pub_localization = 50.0

        # Start the transform fusion thread
        fusion_thread = threading.Thread(target=self.transform_fusion, daemon=True)
        fusion_thread.start()

    def transform_fusion(self):
        global cur_odom_to_baselink, cur_map_to_odom
        rate = 1.0 / self.freq_pub_localization
        while rclpy.ok():
            time.sleep(rate)
            # Copy the latest odometry from global variable
            cur_odom = copy.copy(cur_odom_to_baselink)
            if cur_map_to_odom is not None:
                T_map_to_odom = pose_to_mat(cur_map_to_odom)
            else:
                T_map_to_odom = np.eye(4)
            # Prepare and send transform (from map to camera_init)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'camera_init'
            translation = tf_trans.translation_from_matrix(T_map_to_odom)
            quaternion = tf_trans.quaternion_from_matrix(T_map_to_odom)
            t.transform.translation.x = translation[0]
            t.transform.translation.y = translation[1]
            t.transform.translation.z = translation[2]
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]
            self.br.sendTransform(t)
            # If we have a current odometry, publish localization
            if cur_odom is not None:
                localization = Odometry()
                T_odom_to_base_link = pose_to_mat(cur_odom)
                T_map_to_base_link = np.matmul(T_map_to_odom, T_odom_to_base_link)
                xyz = tf_trans.translation_from_matrix(T_map_to_base_link)
                quat = tf_trans.quaternion_from_matrix(T_map_to_base_link)
                # localization.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
                localization.pose.pose.position.x = xyz[0]
                localization.pose.pose.position.y = xyz[1]
                localization.pose.pose.position.z = xyz[2]
                localization.pose.pose.orientation.x = quat[0]
                localization.pose.pose.orientation.y = quat[1]
                localization.pose.pose.orientation.z = quat[2]
                localization.pose.pose.orientation.w = quat[3]
                localization.twist = cur_odom.twist
                localization.header.stamp = cur_odom.header.stamp
                localization.header.frame_id = 'map'
                localization.child_frame_id = 'body'
                self.pub_localization.publish(localization)

    def cb_save_cur_odom(self, odom_msg):
        global cur_odom_to_baselink
        cur_odom_to_baselink = odom_msg

    def cb_save_map_to_odom(self, odom_msg):
        global cur_map_to_odom
        cur_map_to_odom = odom_msg


def main(args=None):
    rclpy.init(args=args)
    node = TransformFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
