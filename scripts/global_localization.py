#!/usr/bin/env python3
# coding=utf-8

import time
import threading
import rclpy
from rclpy.node import Node
import open3d as o3d
import numpy as np
import tf_transformations as tf_trans

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import ros2_numpy

# Global constants (as before)
MAP_VOXEL_SIZE = 0.1
SCAN_VOXEL_SIZE = 0.1
FREQ_LOCALIZATION = 0.5         # Hz
LOCALIZATION_TH = 0.95
FOV = 6.28                       # radians
FOV_FAR = 30                   # meters

# Global variables for the sake of this example
global_map = None
cur_scan = None
cur_odom = None
T_map_to_odom = np.eye(4)

# (Define your helper functions like pose_to_mat, msg_to_array, voxel_down_sample, etc.)

class FastLioLocalization(Node):
    def __init__(self):
        super().__init__('fast_lio_localization')
        self.initial_pose_received = False
        self.initial_pose = None

        # Create persistent subscriptions
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 1)
        self.create_subscription(PointCloud2, '/cloud_registered', self.cb_save_cur_scan, 1)
        self.create_subscription(Odometry, '/Odometry', self.cb_save_cur_odom, 1)

        # Create publishers
        self.pub_pc_in_map = self.create_publisher(PointCloud2, '/cur_scan_in_map', 1)
        self.pub_submap = self.create_publisher(PointCloud2, '/submap', 1)
        self.pub_map_to_odom = self.create_publisher(Odometry, '/map_to_odom', 1)

        # Timer for periodic localization
        self.create_timer(1.0 / FREQ_LOCALIZATION, self.localization_timer_callback)
        self.get_logger().info('Localization Node Initialized...')

        # Initialize global map (synchronously or via another subscription)
        self.wait_for_global_map()
        self.previous_transformation = None

    def wait_for_global_map(self):
        self.get_logger().warn('Waiting for global map......')
        global global_map
        global_map_msg = None

        def temp_callback(msg):
            nonlocal global_map_msg
            global_map_msg = msg

        temp_sub = self.create_subscription(PointCloud2, '/map', temp_callback, 1)
        while global_map_msg is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        temp_sub.destroy()
        # Initialize global map using your helper function
        global_map = o3d.geometry.PointCloud()
        global_map.points = o3d.utility.Vector3dVector(self.msg_to_array(global_map_msg)[:, :3])
        global_map = self.voxel_down_sample(global_map, MAP_VOXEL_SIZE)
        # point to plane normal estimation
        # global_map.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.get_logger().info('Global map received.')

    def initial_pose_callback(self, msg):
        if not self.initial_pose_received:
            self.initial_pose = self.pose_to_mat(msg)
            self.initial_pose_received = True
            self.get_logger().info('Initial pose received.')

    def localization_timer_callback(self):
        if self.initial_pose_received and cur_scan is not None and cur_odom is not None:
            # Call your global_localization function here.
            # Ensure that global_localization uses the node's methods (like get_logger, publish, etc.) properly.
            self.global_localization(self.initial_pose)
        else:
            self.get_logger().warn('Waiting for initial pose or scan...')
    
    def get_difference(self, T_prev, T_curr):
        R_prev = T_prev[:3, :3]
        R_curr = T_curr[:3, :3]
        p_prev = T_prev[:3, 3]
        p_curr = T_curr[:3, 3]
        angle = np.arccos((np.trace(R_prev @ R_curr.T) - 1) / 2)
        
        translation = np.linalg.norm((R_prev - R_curr) * self.T_odom_to_base_link[:3,3] + (p_prev - p_curr),2)
         
        return angle, translation

    def global_localization(self, pose_estimation):
        # Your implementation of global localization adapted for ROS2.
        # Use a copy of the current scan for thread safety
        import copy
        scan_tobe_mapped = copy.copy(cur_scan)
        # Crop the global map in FOV and run registration (using your helper functions)
        global_map_in_FOV = self.crop_global_map_in_FOV(global_map, pose_estimation, cur_odom)
        tick = time.time()
        if self.previous_transformation is None:
            transformation, _ = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5, max_iteration = 100)
            transformation, fitness = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation, scale=1, max_iteration=100)
            angle = 0
            translation = 0
            fitness = 1
        else:
            # transformation, _ = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)
            transformation, fitness = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=1)
            angle, translation = self.get_difference(self.previous_transformation, transformation)

        tock = time.time()
        self.get_logger().info("Registration time: {}s".format(tock - tick))
        if fitness > LOCALIZATION_TH:
            global T_map_to_odom
            T_map_to_odom = transformation
            map_to_odom = Odometry()
            xyz = tf_trans.translation_from_matrix(T_map_to_odom)
            quat = tf_trans.quaternion_from_matrix(T_map_to_odom)
            map_to_odom.pose.pose.position.x = xyz[0]
            map_to_odom.pose.pose.position.y = xyz[1]
            map_to_odom.pose.pose.position.z = xyz[2]
            map_to_odom.pose.pose.orientation.x = quat[0]
            map_to_odom.pose.pose.orientation.y = quat[1]
            map_to_odom.pose.pose.orientation.z = quat[2]
            map_to_odom.pose.pose.orientation.w = quat[3]
            map_to_odom.header.stamp = cur_odom.header.stamp
            map_to_odom.header.frame_id = 'map'
            self.pub_map_to_odom.publish(map_to_odom)
            self.previous_transformation = transformation
            self.initial_pose = transformation
        else:
            self.get_logger().info("Angle: {}, Translation: {}, Fitness {}".format(angle, translation, fitness))
        
    def cb_save_cur_scan(self, pc_msg):
        global cur_scan
        # Process the incoming point cloud message similarly to your original code.
        pc_msg.header.frame_id = 'camera_init'
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_pc_in_map.publish(pc_msg)
        # Optionally adjust field ordering as needed, then convert to Open3D format.
        pc = self.msg_to_array(pc_msg)
        cur_scan = o3d.geometry.PointCloud()
        cur_scan.points = o3d.utility.Vector3dVector(pc[:, :3])

    def cb_save_cur_odom(self, odom_msg):
        global cur_odom
        cur_odom = odom_msg

    # Dummy implementations for helper methods to illustrate structure:
    def pose_to_mat(self, pose_msg):
        # Convert a PoseWithCovarianceStamped into a transformation matrix
        pos = pose_msg.pose.pose.position
        quat = pose_msg.pose.pose.orientation
        trans = np.array([pos.x, pos.y, pos.z])
        mat = tf_trans.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
        mat[:3, 3] = trans
        return mat

    # def msg_to_array(self, pc_msg):
    #     pc_array = ros2_numpy.numpify(pc_msg)
    #     pc = np.zeros((len(pc_array), 3), dtype=np.float32)
    #     pc[:, 0] = pc_array['x']
    #     pc[:, 1] = pc_array['y']
    #     pc[:, 2] = pc_array['z']
    #     return pc
    
    # Point Cloud data error (I don't know why this error occurs)
    def msg_to_array_custom(self, pc_msg):
        # Check that point_step is as expected
        ps = pc_msg.point_step  # should be 48 bytes
        width = pc_msg.width    # number of points
        total_expected = width * ps
        if len(pc_msg.data) != total_expected:
            print(f"Warning: Expected data length {total_expected}, got {len(pc_msg.data)}")
        
        # Define a dtype for just x, y, z fields (each float32) at the proper offsets.
        # Even though the full point is 48 bytes, we only extract x, y, and z.
        dtype_xyz = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [np.float32, np.float32, np.float32],
            'offsets': [0, 4, 8],
            'itemsize': ps  # entire point size, so numpy skips the rest automatically
        })
        
        # Use np.frombuffer to convert the raw data into our custom array
        pc_array = np.frombuffer(pc_msg.data, dtype=dtype_xyz)
        
        # Optionally, if you want a simple (N,3) float32 array:
        pc_xyz = np.empty((pc_array.shape[0], 3), dtype=np.float32)
        pc_xyz[:, 0] = pc_array['x']
        pc_xyz[:, 1] = pc_array['y']
        pc_xyz[:, 2] = pc_array['z']
        
        return pc_xyz

# Then in your callback, replace the call:
    def msg_to_array(self, pc_msg):
        return self.msg_to_array_custom(pc_msg)

    def voxel_down_sample(self, pcd, voxel_size):
        try:
            return pcd.voxel_down_sample(voxel_size)
        except Exception:
            return o3d.geometry.voxel_down_sample(pcd, voxel_size)

    def registration_at_scale(self, pc_scan, pc_map, initial, scale, max_iteration=20):
        result_icp = o3d.pipelines.registration.registration_icp(
            self.voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale),
            self.voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale),
            1.0 * scale,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration)
        )
        return result_icp.transformation, result_icp.fitness

    def crop_global_map_in_FOV(self, global_map, pose_estimation, cur_odom):
        # Similar implementation as before
        self.T_odom_to_base_link = self.pose_to_mat(cur_odom)
        T_map_to_base_link = np.matmul(pose_estimation, self.T_odom_to_base_link)
        T_base_link_to_map = self.inverse_se3(T_map_to_base_link)
        global_map_in_map = np.array(global_map.points)
        global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
        global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map.T).T
        if FOV > 3.14: # MID360
            # indices = np.where(
            # (global_map_in_base_link[:, 0]**2 + global_map_in_base_link[:, 1]**2 + global_map_in_base_link[:, 2]**2 <= FOV_FAR**2) &
            # (global_map_in_base_link[:, 2] >= -10)  & (global_map_in_base_link[:, 2] <= 10)
            # )
            
            indices = np.where(
                (global_map_in_base_link[:, 0] < FOV_FAR) & (global_map_in_base_link[:, 0] > -FOV_FAR) &
                (global_map_in_base_link[:, 1] < FOV_FAR) & (global_map_in_base_link[:, 1] > -FOV_FAR)
            )
        else:
            indices = np.where(
                (global_map_in_base_link[:, 0] > 0) &
                (global_map_in_base_link[:, 0] < FOV_FAR) &
                (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
            )
        global_map_in_FOV = o3d.geometry.PointCloud()
        global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))
        # Publish FOV point cloud if desired
        header = cur_odom.header
        header.frame_id = 'map'
        points = np.array(global_map_in_FOV.points)[::10]
        data = np.zeros(points.shape[0], dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32)])
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        msg = ros2_numpy.msgify(PointCloud2, data)
        msg.header = header
        self.pub_submap.publish(msg)
        # self.pub_submap.publish(ros2_numpy.msgify(PointCloud2, 
        #                                            np.array(global_map_in_FOV.points)[::10],
        #                                            header=header))
        
        # point to plane normal estimation
        # if len(global_map.normals) > 0:
        #     normals = np.asarray(global_map.normals)[indices, :]
        #     global_map_in_FOV.normals = o3d.utility.Vector3dVector(np.squeeze(normals))
        return global_map_in_FOV

    def inverse_se3(self, trans):
        trans_inverse = np.eye(4)
        trans_inverse[:3, :3] = trans[:3, :3].T
        trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inverse

def main(args=None):
    rclpy.init(args=args)
    node = FastLioLocalization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
