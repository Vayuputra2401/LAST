"""
NTU RGB+D Skeleton File Parser

Parses .skeleton files from NTU RGB+D 120 dataset into numpy arrays.
"""

import numpy as np
from typing import Tuple, Optional
import os


class SkeletonFileParser:
    """
    Parser for NTU RGB+D .skeleton files.
    
    File format:
        Line 1: Total frames
        For each frame:
            - Number of bodies
            - For each body:
                - Body metadata (tracking ID, hand states, etc.)
                - Number of joints (25)
                - For each joint: 3D coords, 2D coords, orientation, tracking state
    """
    
    def __init__(self, num_joints: int = 25, max_bodies: int = 2):
        """
        Args:
            num_joints: Number of joints per skeleton (25 for NTU RGB+D)
            max_bodies: Maximum number of bodies to track per frame
        """
        self.num_joints = num_joints
        self.max_bodies = max_bodies
    
    def parse_file(self, file_path: str) -> Tuple[np.ndarray, dict]:
        """
        Parse a .skeleton file.
        
        Args:
            file_path: Path to .skeleton file
            
        Returns:
            skeleton_data: np.ndarray of shape (T, V, C, M)
                T = number of frames
                V = number of joints (25)
                C = coordinates (3 for x, y, z)
                M = max bodies (2)
            metadata: dict with parsing information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Skeleton file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        line_idx = 0
        
        # Read total frames
        num_frames = int(lines[line_idx].strip())
        line_idx += 1
        
        # Initialize output array: (T, V, C, M)
        skeleton_data = np.zeros((num_frames, self.num_joints, 3, self.max_bodies), dtype=np.float32)
        
        frame_bodies_count = []
        
        for frame_idx in range(num_frames):
            # Read number of bodies in this frame
            num_bodies = int(lines[line_idx].strip())
            line_idx += 1
            
            frame_bodies_count.append(num_bodies)
            
            # Process up to max_bodies
            bodies_to_process = min(num_bodies, self.max_bodies)
            
            for body_idx in range(num_bodies):
                # Read body metadata (1 line)
                body_info = lines[line_idx].strip().split()
                line_idx += 1
                
                # Read number of joints
                num_joints = int(lines[line_idx].strip())
                line_idx += 1
                
                # Only store if within max_bodies limit
                if body_idx < self.max_bodies:
                    for joint_idx in range(num_joints):
                        joint_data = lines[line_idx].strip().split()
                        line_idx += 1
                        
                        # Parse 3D coordinates (first 3 values)
                        x = float(joint_data[0])
                        y = float(joint_data[1])
                        z = float(joint_data[2])
                        
                        skeleton_data[frame_idx, joint_idx, :, body_idx] = [x, y, z]
                else:
                    # Skip joints for bodies beyond max_bodies
                    for joint_idx in range(num_joints):
                        line_idx += 1
        
        metadata = {
            'num_frames': num_frames,
            'num_joints': self.num_joints,
            'max_bodies': self.max_bodies,
            'frame_bodies_count': frame_bodies_count,
            'file_path': file_path
        }
        
        return skeleton_data, metadata
    
    def parse_file_simple(self, file_path: str) -> np.ndarray:
        """
        Simplified parsing: returns only first body with shape (T, V, C).
        
        Args:
            file_path: Path to .skeleton file
            
        Returns:
            skeleton_data: np.ndarray of shape (T, V, C)
                T = number of frames
                V = number of joints (25)
                C = coordinates (3 for x, y, z)
        """
        full_data, metadata = self.parse_file(file_path)
        # Return first body only: (T, V, C, M) -> (T, V, C)
        return full_data[:, :, :, 0]
    
    @staticmethod
    def extract_label_from_filename(file_path: str) -> int:
        """
        Extract action label from NTU RGB+D filename.
        
        Filename format: S<setup>C<camera>P<person>R<replication>A<action>.skeleton
        
        Args:
            file_path: Path to skeleton file
            
        Returns:
            action_label: Action class (0-119 for NTU RGB+D 120)
        """
        filename = os.path.basename(file_path)
        # Extract action number from filename (e.g., 'A013' -> 13)
        action_str = filename.split('A')[-1].split('.')[0]
        action_label = int(action_str) - 1  # Convert to 0-indexed
        return action_label
    
    @staticmethod
    def extract_metadata_from_filename(file_path: str) -> dict:
        """
        Extract all metadata from NTU RGB+D filename.
        
        Args:
            file_path: Path to skeleton file
            
        Returns:
            metadata: dict with setup, camera, person, replication, action
        """
        filename = os.path.basename(file_path)
        # Remove .skeleton extension
        filename = filename.replace('.skeleton', '')
        
        # Parse: S001C001P001R001A001
        parts = filename.split('C')
        setup = int(parts[0].replace('S', ''))
        
        parts = parts[1].split('P')
        camera = int(parts[0])
        
        parts = parts[1].split('R')
        person = int(parts[0])
        
        parts = parts[1].split('A')
        replication = int(parts[0])
        action = int(parts[1])
        
        return {
            'setup': setup,
            'camera': camera,
            'person': person,
            'replication': replication,
            'action': action - 1  # 0-indexed
        }
