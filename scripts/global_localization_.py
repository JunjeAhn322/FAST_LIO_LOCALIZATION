#!/usr/bin/env python3
# coding=utf-8

import time
import copy
import threading
import rclpy
import rclpy.logging
import open3d as o3d
import numpy as np
import tf_transformations as tf_trans

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import ros2_numpy

# Global variables for the sake of this example
global_map = None
cur_scan = None
cur_odom = None
T_map_to_odom = np.eye(4)
initialized = False

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

def msg_to_array(pc_msg):# Debug print to check conversion result
    pc_array = ros2_numpy.numpify(pc_msg)
    if pc_array is None:
        raise ValueError("Conversion failed: received None. Check the PointCloud2 message format and data.")
    pc = np.zeros([len(pc_array), 3])
    pc[:, 0] = pc_array['x']
    pc[:, 1] = pc_array['y']
    pc[:, 2] = pc_array['z']
    return pc

def registration_at_scale(pc_scan, pc_map, initial, scale):
    result_icp = o3d.pipelines.registration.registration_icp(
        voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale), voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale),
        1.0 * scale, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )
    
    return result_icp.transformation, result_icp.fitness

def inverse_se3(trans):
    trans_inv = np.eye(4)
    trans_inv[:3, :3] = trans[:3, :3].T
    trans_inv[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
    return trans_inv

def publish_point_cloud(publisher, header, pc):
    data = np.zeros(len(pc), dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32),
    ])
    data['x'] = pc[:, 0]
    data['y'] = pc[:, 1]
    data['z'] = pc[:, 2]
    if pc.shape[1] == 4:
        data['intensity'] = pc[:, 3]
    msg = ros2_numpy.msgify(PointCloud2, data)
    msg.header = header
    publisher.publish(msg)
    
def crop_global_map_in_FOV(global_map, pose_estimation, cur_odom):
    T_odom_to_base_link = pose_to_mat(cur_odom)
    T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
    T_base_link_to_map = inverse_se3(T_map_to_base_link)
    
    global_map_in_map = np.array(global_map.points)
    global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
    global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map.T).T
    
    if FOV > 3.14: # if LiDAR's FOV is more than 180 degrees like MID-360
        indices = np.where(
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    else: # if LiDAR's FOV is less than 180 degrees like AVIA
        indices = np.where(
            (global_map_in_base_link[:, 0] > 0) &
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    global_map_in_FOV = o3d.geometry.PointCloud()
    global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))
    
    header = cur_odom.header
    header.frame_id = 'map'
    publish_point_cloud(pub_submap, header, np.array(global_map_in_FOV.points)[::10])
    
    return global_map_in_FOV

def global_localization(pose_estimation):
    global global_map, cur_scan, cur_odom, T_map_to_odom
    
    print('Global localization by scan-to-map matching......')
    
    scan_tobe_mapped = copy.copy(cur_scan)
    
    tic = time.time()
    
    global_map_in_FOV = crop_global_map_in_FOV(global_map, pose_estimation, cur_odom)
    
    transformation, _ = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)
    
    transformation, fitness = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation, scale=1)
    
    toc = time.time()
    print("Time: {.3f} seconds".format(toc - tic))
    print("")
    
    if fitness > LOCALIZATION_TH:
        T_map_to_odom = transformation
        
        map_to_odom = transformation
        
        map_to_odom = Odometry()
        xyz = tf_trans.translation_from_matrix(T_map_to_odom)
        quat = tf_trans.quaternion_from_matrix(T_map_to_odom)
        map_to_odom.pose.pose = Pose(
            position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        )
        map_to_odom.header.stamp = cur_odom.header.stamp
        map_to_odom.header.frame_id = 'map'
        pub_map_to_odom.publish(map_to_odom)
        return True
    else:
        print("Not match!!!!")
        print("{}".format(transformation))
        print("fitness score:{}".format(fitness))
        return False
    
def voxel_down_sample(pcd, voxel_size):
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    except:
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd_down

def initialize_global_map(pc_msg):
    global global_map
    
    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
    global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
    print('Global map received.')
    
def cb_save_cur_scan(pc_msg):
    global cur_scan
    
    pc_msg.header.frame_id = 'camera_init'
    pc_msg.header.stamp = rclpy.time.Time().to_msg()
    pub_pc_in_map.publish(pc_msg)
    
    pc_msg.fields = [pc_msg.fields[0], pc_msg.fields[1], pc_msg.fields[2],
                     pc_msg.fields[4], pc_msg.fields[5], pc_msg.fields[6],
                     pc_msg.fields[3], pc_msg.fields[7]]
    pc = msg_to_array(pc_msg)
    
    cur_scan = o3d.geometry.PointCloud()
    cur_scan.points = o3d.utility.Vector3dVector(pc[:, :3])
    
def cb_save_cur_odom(odom_msg):
    global cur_odom
    cur_odom = odom_msg
    
def thread_localization():
    global T_map_to_odom
    while True:
        time.sleep(1.0 / FREQ_LOCALIZATION)
        global_localization(T_map_to_odom)
        
# def wait_for_map(topic, node, qos_profile, msg_type, timeout = None):
#     msg = None
#     event = threading.Event()
#     def callback(m):
#         nonlocal msg
#         msg = m
#         event.set()
        
#     sub = node.create_subscription(msg_type, topic, callback, qos_profile)
#     success = event.wait(timeout)
#     node.destroy_subscription(sub)
    
#     if success:
#         return msg
#     else:
#         node.get_logger().warn('Timeout while waiting for message on topic {}'.format(topic))
#         return None
  
def wait_for_global_map(node):
    print('Waiting for global map......')
    global global_map
    global_map_msg = None

    def temp_callback(msg):
        nonlocal global_map_msg
        global_map_msg = msg

    temp_sub = node.create_subscription(PointCloud2, '/map', temp_callback, 1)
    while global_map_msg is None:
        rclpy.spin_once(node,timeout_sec=0.1)
    temp_sub.destroy()
    # Initialize global map using your helper function
    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(msg_to_array(global_map_msg)[:, :3])
    global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
    print('Global map received.')

# def main():
#     MAP_VOXEL_SIZE = 0.4
#     SCAN_VOXEL_SIZE = 0.1

#     FREQ_LOCALIZATION = 0.5
#     LOCALIZATION_TH = 0.95

#     FOV = 6.28
#     FOV_FAR = 150

#     rclpy.init()
#     node = rclpy.create_node('fast_lio_localization')
#     print('Localization Node Initialized...')

#     qos_profile = rclpy.qos.QoSProfile(
#         history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
#         depth=1,
#         reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
#     )

#     pub_pc_in_map = node.create_publisher(PointCloud2, '/cur_scan_in_map', qos_profile)
#     pub_submap = node.create_publisher(PointCloud2, '/submap', qos_profile)
#     pub_map_to_odom = node.create_publisher(Odometry, '/map_to_odom', qos_profile)

#     node.create_subscription(PointCloud2, '/cloud_registered', cb_save_cur_scan, qos_profile)
#     node.create_subscription(Odometry, '/Odometry', cb_save_cur_odom, qos_profile)

#     # Start a separate thread to spin the node so that callbacks are processed.
#     spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     spin_thread.start()

#     print('Waiting for global map......')
#     map_msg = wait_for_global_map(node)
#     initialize_global_map(map_msg)

#     # Global variable 'initialized' should be declared outside (e.g., at module scope)
#     while not initialized:
#         print('Waiting for initial pose......')
#         pose_msg = wait_for_map('/initialpose', node, qos_profile, PoseWithCovarianceStamped, timeout=2)
#         if pose_msg is not None:
#             initial_pose = pose_to_mat(pose_msg)
#             if cur_scan:
#                 initialized = global_localization(initial_pose)
#             else:
#                 print('No scan data received yet.')
#         else:
#             print('Initial pose not received.')

#     print('Initialization successfully!!!!!!!!!!')
#     t = threading.Thread(target=thread_localization, args=())
#     t.start()

#     rclpy.spin(node)

# if __name__ == '__main__':
#     main()
    
def main():
    MAP_VOXEL_SIZE = 0.4
    SCAN_VOXEL_SIZE = 0.1
    
    FREQ_LOCALIZATION = 0.5
    LOCALIZATION_TH = 0.95
    
    FOV = 6.28
    FOV_FAR = 150
    
    rclpy.init()
    node = rclpy.create_node('fast_lio_localization')
    print('Localization Node Initialized...')
    
    qos_profile = rclpy.qos.QoSProfile(
        history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
        depth=1, 
        reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)

    pub_pc_in_map = node.create_publisher(PointCloud2, '/cur_scan_in_map', qos_profile)
    pub_submap = node.create_publisher(PointCloud2, '/submap', qos_profile)
    pub_map_to_odom = node.create_publisher(Odometry, '/map_to_odom', qos_profile)
    
    node.create_subscription(PointCloud2, '/cloud_registered', cb_save_cur_scan, qos_profile)
    node.create_subscription(Odometry, '/Odometry', cb_save_cur_odom, qos_profile)
    
    print('Waiting for global map......')
    initialize_global_map(wait_for_global_map(node))
    
    while not initialized:
        print('Waiting for initial pose......')
        
        pose_msg = wait_for_map('/initialpose', node, qos_profile, PoseWithCovarianceStamped)
        initial_pose = pose_to_mat(pose_msg)
        if cur_scan:
            initialized = global_localization(initial_pose)
        else:
            print('No scan data received yet.')
            
    print('Initialization successfully!!!!!!!!!!')
    t= threading.Thread(target = thread_localization, args = ())
    t.start()
    
    rclpy.spin(node)
        
if __name__ == '__main__':
    main()