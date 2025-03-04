#!/usr/bin/env python3
# coding=utf-8

import copy
import threading
import time

import numpy as np
import rclpy
import tf_transformations as tf_trans  # pip install tf-transformations if needed
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry

# Global variables (you may encapsulate these as members of your Node subclass)
cur_odom_to_baselink = None
cur_map_to_odom = None
FREQ_PUB_LOCALIZATION = 50.0

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

def transform_fusion(node, pub_localization):
    global cur_odom_to_baselink, cur_map_to_odom
    
    br = TransformBroadcaster(node)
    while True:
        time.sleep(1 / FREQ_PUB_LOCALIZATION)
        
        cur_odom = copy.copy(cur_odom_to_baselink)
        if cur_map_to_odom is not None:
            T_map_to_odom = pose_to_mat(cur_map_to_odom)
        else:
            T_map_to_odom = np.eye(4)
            
        t = TransformStamped()
        t.header.stamp = node.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera_init'
        
        translation = tf_trans.translation_from_matrix(T_map_to_odom)
        rotation = tf_trans.quaternion_from_matrix(T_map_to_odom)
        
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]
        
        br.sendTransform(t)
        
        if cur_odom is not None:
            localization = Odometry()
            T_odom_to_base_link = pose_to_mat(cur_odom)
            T_map_to_base_link = np.matmul(T_map_to_odom, T_odom_to_base_link)
            xyz = tf_trans.translation_from_matrix(T_map_to_base_link)
            quat = tf_trans.quaternion_from_matrix(T_map_to_base_link)
            localization.pose.pose = Pose(
                position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            )
            localization.twist = cur_odom.twist
            
            localization.header.stamp = cur_odom.header.stamp
            localization.header.frame_id = 'map'
            localization.child_frame_id = 'body'
            pub_localization.publish(localization)
            
def cb_save_cur_odom(odom_msg):
    global cur_odom_to_baselink
    cur_odom_to_baselink = odom_msg
    
def cb_save_map_to_odom(odom_msg):
    global cur_map_to_odom
    cur_map_to_odom = odom_msg
    
def main(args=None):
    qos_profile = rclpy.qos.QoSProfile(
        history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
        depth=1, 
        reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
    rclpy.init(args=args)
    node = rclpy.create_node('transform_fusion')
    rclpy.logging.get_logger('transform_fusion node inited ...')
    
    node.create_subscription(Odometry, '/Odometry', cb_save_cur_odom, qos_profile)
    node.create_subscription(Odometry, '/map_to_odom', cb_save_map_to_odom, qos_profile)
    
    pub_localization = node.create_publisher(Odometry, '/localization', qos_profile)
    
    t= threading.Thread(target = transform_fusion, args = (node, pub_localization, ))
    t.start()

    rclpy.spin(node)

if __name__ == '__main__':
    main()