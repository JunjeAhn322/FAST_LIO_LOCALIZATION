#!/usr/bin/env python3
# coding=utf-8

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import tf_transformations  # For converting rotation matrix to quaternion
from livox_ros_driver2.msg import CustomMsg  # Adjust if needed for your custom message
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped

import tf2_ros
from rclpy.duration import Duration

class LivoxFPFHLocalization(Node):
    def __init__(self):
        super().__init__('livox_fpfh_localization')
        # Load the map from a parameter
        self.declare_parameter("map_file", "/home/raibo-con/ros2_ws/src/FAST_LIO/PCD/surrounding_every_frame.pcd")
        map_file = self.get_parameter("map_file").value
        self.map_pcd = o3d.io.read_point_cloud(map_file)
        self.get_logger().info(f"PCD map loaded successfully from {map_file}")
        
        # Create a TF2 buffer and listener for frame transformation.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Latest IMU data holder
        self.latest_imu = None
        
        # Subscribe to Livox lidar messages
        self.subscription = self.create_subscription(
            CustomMsg,
            '/livox/lidar',
            self.lidar_callback,
            10
        )
        
        # Subscribe to Livox IMU messages
        self.imu_subscription = self.create_subscription(
            Imu,
            '/livox/imu',
            self.imu_callback,
            10
        )
        
        # Publisher for initial pose estimation
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    
    def imu_callback(self, msg):
        self.latest_imu = msg
        self.get_logger().info("IMU data received.")
    
    def get_initial_guess_from_imu(self):
        """
        Constructs a rotation matrix from the latest IMU data using roll and pitch,
        while setting yaw to zero (since the IMU yaw is arbitrary at startup).
        """
        if self.latest_imu is None:
            return np.eye(4)
        q = [
            self.latest_imu.orientation.x,
            self.latest_imu.orientation.y,
            self.latest_imu.orientation.z,
            self.latest_imu.orientation.w
        ]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(q)
        # Override yaw to 0 for alignment with the map's coordinate system.
        new_quat = tf_transformations.quaternion_from_euler(roll, pitch, 0.0)
        T = tf_transformations.quaternion_matrix(new_quat)
        return T
    
    def debug_print_cloud_info(self, pcd, label=""):
        points = np.asarray(pcd.points)
        if points.size == 0:
            self.get_logger().warn(f"{label} point cloud is empty!")
            return
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        self.get_logger().info(
            f"{label} point cloud: {len(points)} points | Bounds: min {min_bound}, max {max_bound}"
        )
    
    def visualize_clouds(self, source, target):
        source_temp = source.clone()
        target_temp = target.clone()
        source_temp.paint_uniform_color([1, 0, 0])  # Red: scan
        target_temp.paint_uniform_color([0, 1, 0])  # Green: map (or cropped map)
        o3d.visualization.draw_geometries([source_temp, target_temp])
    
    def transform_scan_to_map(self, scan_pcd):
        """
        Combines TF translation with IMU-derived rotation (roll and pitch only)
        to roughly align the scan with the map frame.
        """
        try:
            now = self.get_clock().now().to_msg()
            transform_stamped = self.tf_buffer.lookup_transform(
                "map", "livox_frame", now, Duration(seconds=1.0)
            )
            t = transform_stamped.transform.translation
            trans = [t.x, t.y, t.z]
        except Exception as e:
            self.get_logger().warn(f"Failed to lookup transform from livox_frame to map: {e}")
            trans = [0, 0, 0]
        
        if self.latest_imu is not None:
            imu_T = self.get_initial_guess_from_imu()
            imu_T[0:3, 3] = trans  # Use the translation from TF
            self.get_logger().info(f"Using IMU rotation with translation:\n{imu_T}")
            scan_pcd.transform(imu_T)
        else:
            self.get_logger().info("No IMU data available, using identity transformation.")
        return scan_pcd

    def crop_map(self, map_pcd, scan_pcd, margin=5.0):
        scan_points = np.asarray(scan_pcd.points)
        if scan_points.size == 0:
            self.get_logger().warn("Scan point cloud is empty during cropping!")
            return map_pcd
        min_bound = scan_points.min(axis=0) - margin
        max_bound = scan_points.max(axis=0) + margin
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        cropped_map = map_pcd.crop(bbox)
        self.debug_print_cloud_info(cropped_map, "Cropped Map")
        return cropped_map

    def preprocess_point_cloud(self, pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        self.debug_print_cloud_info(pcd_down, "Downsampled")
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        radius_feature = voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, fpfh

    def initial_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5  
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999)
        )
        self.get_logger().info(f"RANSAC registration fitness: {result.fitness:.3f}")
        if result.fitness < 0.3:
            return None
        return result.transformation

    def refine_registration(self, source_down, target_down, init_transformation, voxel_size):
        distance_threshold = voxel_size * 0.4
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, distance_threshold, init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        self.get_logger().info(f"ICP refinement fitness: {result.fitness:.3f}")
        if result.fitness < 0.3:
            return None
        return result.transformation

    def lidar_callback(self, msg):
        self.get_logger().info(f"Received Livox lidar message with frame_id: {msg.header.frame_id}")
        points = np.array([[p.x, p.y, p.z] for p in msg.points])
        if points.size == 0:
            self.get_logger().warn("Received lidar message with no points!")
            return
        self.get_logger().info(f"First 3 points from message: {points[:3]}")
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(points)
        self.debug_print_cloud_info(scan_pcd, "Scan original")
        
        # Transform scan to the map frame using TF translation and IMU roll/pitch.
        scan_pcd = self.transform_scan_to_map(scan_pcd)
        
        # Preprocess the scan
        voxel_size = 1.0  # Tuning parameter (e.g., 0.5 for sparser scans)
        scan_down, scan_fpfh = self.preprocess_point_cloud(scan_pcd, voxel_size)
        
        # Crop the map around the scan to improve registration success
        cropped_map = self.crop_map(self.map_pcd, scan_down, margin=5.0)
        
        # Preprocess the cropped map
        map_down, map_fpfh = self.preprocess_point_cloud(cropped_map, voxel_size)
        
        # (Optional) Visualize the downsampled clouds to verify overlap:
        # self.visualize_clouds(scan_down, map_down)
        
        init_trans = self.initial_registration(scan_down, map_down, scan_fpfh, map_fpfh, voxel_size)
        if init_trans is None:
            self.get_logger().warn("Initial registration failed!")
            return
        
        self.get_logger().info("Initial registration complete. Refining with ICP...")
        refined_trans = self.refine_registration(scan_down, map_down, init_trans, voxel_size)
        if refined_trans is None:
            self.get_logger().warn("ICP refinement failed!")
            return
        
        self.get_logger().info("Registration Successful!")
        self.publish_initial_pose(refined_trans)
    
    def publish_initial_pose(self, transformation):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = transformation[0, 3]
        pose_msg.pose.pose.position.y = transformation[1, 3]
        pose_msg.pose.pose.position.z = transformation[2, 3]
        quat = tf_transformations.quaternion_from_matrix(transformation)
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]
        self.get_logger().info(
            "Published initial pose: position: x: {:.3f}, y: {:.3f}, z: {:.3f} | "
            "orientation (quat): x: {:.3f}, y: {:.3f}, z: {:.3f}, w: {:.3f}".format(
                pose_msg.pose.pose.position.x,
                pose_msg.pose.pose.position.y,
                pose_msg.pose.pose.position.z,
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w
            )
        )
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LivoxFPFHLocalization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
