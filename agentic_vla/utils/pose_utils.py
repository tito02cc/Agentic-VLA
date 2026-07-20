"""Core module implementation."""

import numpy as np
from typing import Tuple, List, Optional


class PoseUtils:


    """PoseUtils class."""
    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """quaternion_to_rotation_matrix function."""
        qx, qy, qz, qw = q
        
        # Note

        n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if n > 0:
            qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
            
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
        
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """rotation_matrix_to_quaternion function."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
            
        return np.array([qx, qy, qz, qw])
        
    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """euler_to_quaternion function."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw])
        
    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
        """quaternion_to_euler function."""
        qx, qy, qz, qw = q
        
        # roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
            
        # yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
        
    @staticmethod
    def compose_pose(
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> np.ndarray:
        """compose_pose function."""
        T = np.eye(4)
        T[:3, :3] = PoseUtils.quaternion_to_rotation_matrix(quaternion)
        T[:3, 3] = position
        return T
        
    @staticmethod
    def decompose_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """decompose_pose function."""
        position = T[:3, 3]
        quaternion = PoseUtils.rotation_matrix_to_quaternion(T[:3, :3])
        return position, quaternion
        
    @staticmethod
    def transform_pose(
        pose: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """transform_pose function."""
        return transform @ pose
        
    @staticmethod
    def interpolate_poses(
        pose1: np.ndarray,
        pose2: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """interpolate_poses function."""
        # Note

        pos1 = pose1[:3, 3]
        pos2 = pose2[:3, 3]
        pos = (1 - t) * pos1 + t * pos2
        
        # Note

        q1 = PoseUtils.rotation_matrix_to_quaternion(pose1[:3, :3])
        q2 = PoseUtils.rotation_matrix_to_quaternion(pose2[:3, :3])
        q = PoseUtils.slerp(q1, q2, t)
        
        return PoseUtils.compose_pose(pos, q)
        
    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """slerp function."""
        # Note

        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
            
        if dot > 0.9995:
            # Note

            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
            
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        
        return s1 * q1 + s2 * q2
        
    @staticmethod
    def compute_approach_vector(quaternion: np.ndarray) -> np.ndarray:
        """compute_approach_vector function."""
        R = PoseUtils.quaternion_to_rotation_matrix(quaternion)
        return R[:, 2]  # Note
        
    @staticmethod
    def generate_approach_pose(
        grasp_position: np.ndarray,
        grasp_quaternion: np.ndarray,
        approach_distance: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """generate_approach_pose function."""
        approach_vector = PoseUtils.compute_approach_vector(grasp_quaternion)
        approach_position = grasp_position - approach_distance * approach_vector
        return approach_position, grasp_quaternion
        
    @staticmethod
    def adapt_pose_to_object(
        reference_pose: np.ndarray,
        reference_object_pose: np.ndarray,
        current_object_pose: np.ndarray,
    ) -> np.ndarray:
        """adapt_pose_to_object function."""
        # Note

        ref_obj_inv = np.linalg.inv(reference_object_pose)
        grasp_in_obj_frame = ref_obj_inv @ reference_pose
        
        # Note

        adapted_pose = current_object_pose @ grasp_in_obj_frame
        
        return adapted_pose
