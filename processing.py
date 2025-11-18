import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.spatial.distance import euclidean
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for production
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from enum import Enum

class MovementThresholds:
    """Configuration class containing all thresholds for movement detection and analysis"""
    
    # Movement detection thresholds
    ANGLE_CHANGE_THRESHOLD = 15.0  # degrees - minimum angle change to consider movement
    VELOCITY_THRESHOLD = 0.1  # m/s - base velocity threshold for movement detection
    ACCELERATION_THRESHOLD = 0.05  # m/s² - acceleration threshold for movement phases
    
    # Movement phase detection thresholds
    REST_VELOCITY_FACTOR = 0.3  # factor of velocity threshold for rest phase
    MIN_MOVEMENT_DURATION = 10  # frames - minimum duration for valid movement segment
    
    # Velocity calculation parameters
    DEFAULT_FPS = 60  # frames per second for time calculations
    DEFAULT_DT = 1/60  # time step between frames
    
    # Smoothing parameters
    MIN_WINDOW_SIZE = 3  # minimum window size for smoothing
    SAVGOL_POLY_ORDER = 2  # polynomial order for Savitzky-Golay filter
    VELOCITY_SMOOTHING_WINDOW = 21  # window size for velocity smoothing
    
    # Movement classification thresholds
    class Reach:
        MIN_DISPLACEMENT = 0.2  # m - minimum displacement for reach
        MIN_ELBOW_RANGE = 30  # degrees - minimum elbow angle range
        MIN_VELOCITY = 0.1  # m/s - minimum peak velocity
    
    class Push:
        MIN_VELOCITY = 0.2  # m/s - minimum peak velocity for push
        MIN_FORWARD_LATERAL = 0.05  # m - minimum forward/lateral movement
        MIN_ELBOW_RANGE = 15  # degrees - minimum elbow extension
        MIN_DISPLACEMENT = 0.1  # m - minimum overall displacement
    
    class Lift:
        MIN_VERTICAL_MOVEMENT = 0.1  # m - minimum upward movement
        MIN_SHOULDER_RANGE = 20  # degrees - minimum shoulder angle range
    
    class Pull:
        MIN_BACKWARD_MOVEMENT = -0.1  # m - minimum backward movement (negative forward)
        MIN_ELBOW_RANGE = 25  # degrees - minimum elbow flexion range
        MIN_VELOCITY = 0.08  # m/s - minimum velocity for pull
    
    class Carry:
        MIN_VELOCITY = 0.05  # m/s - minimum sustained velocity
        MAX_ELBOW_RANGE = 20  # degrees - maximum elbow range (stable grip)
        MAX_SHOULDER_RANGE = 15  # degrees - maximum shoulder range (stable position)
        MIN_HORIZONTAL_MOVEMENT = 0.1  # m - minimum lateral/forward movement
    
    class Squat:
        MIN_VERTICAL_HIP_MOVEMENT = -0.15  # m - minimum downward hip movement (negative Z)
        MIN_KNEE_BEND = 30  # degrees - minimum knee angle change
        MAX_FORWARD_LEAN = 0.2  # m - maximum forward upper body lean
        MIN_VELOCITY = 0.05  # m/s - minimum movement velocity
    
    class LowerBackBend:
        MIN_FORWARD_SPINE_MOVEMENT = 0.15  # m - minimum forward spine movement
        MIN_VERTICAL_DROP = -0.1  # m - minimum vertical drop of upper spine
        MAX_HIP_VERTICAL_MOVEMENT = -0.05  # m - hip stays relatively stable
        MIN_VELOCITY = 0.05  # m/s - minimum movement velocity
    
    class Walk:
        MIN_HORIZONTAL_DISPLACEMENT = 0.3  # m - minimum horizontal hip movement
        MIN_ALTERNATING_STEPS = 2  # minimum number of alternating leg movements
        MIN_LEG_CYCLE_DURATION = 0.5  # seconds - minimum duration for one step cycle
        MAX_LEG_CYCLE_DURATION = 2.0  # seconds - maximum duration for one step cycle
        MIN_VELOCITY = 0.1  # m/s - minimum movement velocity
        MIN_KNEE_FLEXION_RANGE = 15  # degrees - minimum knee angle variation per leg
    
    class Twist:
        MIN_SPINE_ROTATION = 15  # degrees - minimum spine rotation angle change
        MAX_HIP_HORIZONTAL_DISPLACEMENT = 0.15  # m - hips stay relatively stable
        MIN_SHOULDER_ROTATION_DIFF = 10  # degrees - difference between shoulder rotations
        MIN_VELOCITY = 0.05  # m/s - minimum movement velocity
    
    class Rotate:
        MIN_SPINE_ROTATION = 20  # degrees - minimum spine rotation angle change
        MIN_HIP_HORIZONTAL_DISPLACEMENT = 0.2  # m - body turns while moving
        MIN_VELOCITY = 0.1  # m/s - minimum movement velocity
    
    # Adaptive threshold factors
    VELOCITY_ADAPTIVE_MEAN_FACTOR = 1.0  # factor of mean velocity for adaptive threshold
    VELOCITY_ADAPTIVE_STD_FACTOR = 0.5  # factor of std deviation for adaptive threshold
    
    # Coordination analysis thresholds
    SYNC_THRESHOLD = 1.0  # seconds - maximum time gap for coordinated movements
    HAND_DOMINANCE_FACTOR = 1.2  # factor for determining hand dominance
    
    # Data validation thresholds
    MIN_FRAMES_FOR_ANALYSIS = 5  # minimum frames required for analysis
    MAX_COORDINATE_VALUE = 10.0  # m - maximum reasonable coordinate value
    MIN_COORDINATE_VALUE = -10.0  # m - minimum reasonable coordinate value

class MovementType(Enum):
    """Enumeration of different movement types"""
    REACH = "reach"
    PUSH = "push" 
    PULL = "pull"
    LIFT = "lift"
    CARRY = "carry"
    SQUAT = "squat"
    LOWER_BACK_BEND = "lower_back_bend"
    WALK = "walk"
    TWIST = "twist"
    ROTATE = "rotate"
    UNKNOWN = "unknown"

class MovementPhase(Enum):
    """Enumeration of movement phases"""
    REST = "rest"
    INITIATION = "initiation"
    EXECUTION = "execution"
    COMPLETION = "completion"

class MovementIdentifier:
    """
    A class to identify and analyze human movements from joint coordinate data.
    Uses decision tree logic to classify movements based on biomechanical patterns.
    """
    
    def __init__(self):
        self.data = None
        self.angles_df = None
        self.movement_phases = None
        
        # Thresholds for movement detection (can be tuned)
        self.angle_change_threshold = 15.0  # degrees
        self.velocity_threshold = 0.1  # m/s
        self.acceleration_threshold = 0.05  # m/s²
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load movement data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            # Clear cached body movement segments when new data is loaded
            self._cached_body_segments = None
            print(f"Loaded data with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def calculate_angle_3d(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 between points p1, p2, p3 in 3D space
        p1, p2, p3 are numpy arrays of shape (3,) representing [x, y, z] coordinates
        """
        # Create vectors from p2 to p1 and p2 to p3
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0
        
        # Calculate angle in radians
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def get_joint_coordinates(self, joint_name: str, side: str = None) -> np.ndarray:
        """
        Extract joint coordinates from the dataframe
        Args:
            joint_name: Name of the joint (e.g., 'shoulder', 'elbow', 'wrist')
            side: 'left' or 'right' (None for spine joints)
        Returns:
            numpy array of shape (n_frames, 3) with [x, y, z] coordinates
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Handle special cases for spine joints
        if joint_name in ['upper_spine', 'mid_spine', 'low_spine']:
            # Spine joint coordinates use pattern: spine_{position}_x/y/z
            spine_pos = joint_name.split('_')[0]  # 'upper', 'mid', or 'low'
            coords = self.data[[f'spine_{spine_pos}_x', f'spine_{spine_pos}_y', f'spine_{spine_pos}_z']].values
        else:
            # Regular joints with left/right variants
            if side is None:
                raise ValueError(f"Side must be specified for joint: {joint_name}")
            coords = self.data[[f'{side}_{joint_name}_x', f'{side}_{joint_name}_y', f'{side}_{joint_name}_z']].values
            
        return coords
    
    def calculate_joint_angles(self, side: str = 'left') -> Dict[str, np.ndarray]:
        """
        Calculate various joint angles for movement analysis
        Args:
            side: 'left' or 'right'
        Returns:
            Dictionary containing angle arrays for different joints
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        angles = {}
        n_frames = len(self.data)
        
        # Get joint coordinates
        try:
            shoulder_coords = self.get_joint_coordinates('shoulder', side)
            elbow_coords = self.get_joint_coordinates('elbow', side)
            wrist_coords = self.get_joint_coordinates('wrist', side)
            upper_spine_coords = self.get_joint_coordinates('upper_spine')
        except KeyError as e:
            print(f"Missing joint coordinates: {e}")
            return {}
        
        # Calculate elbow angle (shoulder-elbow-wrist)
        elbow_angles = []
        for i in range(n_frames):
            angle = self.calculate_angle_3d(
                shoulder_coords[i], elbow_coords[i], wrist_coords[i]
            )
            elbow_angles.append(angle)
        angles[f'{side}_elbow_angle'] = np.array(elbow_angles)
        
        # Calculate shoulder angle (upper_spine-shoulder-elbow) 
        shoulder_angles = []
        for i in range(n_frames):
            angle = self.calculate_angle_3d(
                upper_spine_coords[i], shoulder_coords[i], elbow_coords[i]
            )
            shoulder_angles.append(angle)
        angles[f'{side}_shoulder_angle'] = np.array(shoulder_angles)
        
        return angles
    
    def calculate_joint_velocities(self, coordinates: np.ndarray, dt: float = 1/60) -> np.ndarray:
        """
        Calculate joint velocities from position data
        Args:
            coordinates: Joint coordinates array (n_frames, 3)
            dt: Time step between frames (default 60 FPS)
        Returns:
            Velocity magnitudes array
        """
        velocities = np.zeros(len(coordinates))
        
        for i in range(1, len(coordinates)):
            # Calculate 3D velocity magnitude
            velocity_vector = (coordinates[i] - coordinates[i-1]) / dt
            velocities[i] = np.linalg.norm(velocity_vector)
            
        return velocities
    
    def calculate_joint_accelerations(self, velocities: np.ndarray, dt: float = 1/60) -> np.ndarray:
        """
        Calculate joint accelerations from velocity data
        Args:
            velocities: Velocity magnitudes array
            dt: Time step between frames
        Returns:
            Acceleration magnitudes array  
        """
        accelerations = np.zeros(len(velocities))
        
        for i in range(1, len(velocities)):
            accelerations[i] = (velocities[i] - velocities[i-1]) / dt
            
        return accelerations
    
    def detect_movement_phases(self, joint_coords: np.ndarray, velocities: np.ndarray) -> List[MovementPhase]:
        """
        Detect movement phases based on velocity patterns
        Args:
            joint_coords: Joint coordinates over time
            velocities: Joint velocities over time
        Returns:
            List of movement phases for each frame
        """
        phases = []
        n_frames = len(velocities)
        
        # Smooth velocities to reduce noise
        if n_frames > 10:
            window_size = min(5, n_frames // 10)
            smoothed_velocities = signal.savgol_filter(velocities, window_size, 2)
        else:
            smoothed_velocities = velocities
            
        for i in range(n_frames):
            if smoothed_velocities[i] < self.velocity_threshold * 0.3:
                phases.append(MovementPhase.REST)
            elif i < n_frames - 1 and smoothed_velocities[i] < self.velocity_threshold:
                # Check if velocity is increasing (initiation)
                if i < n_frames - 2 and smoothed_velocities[i+1] > smoothed_velocities[i]:
                    phases.append(MovementPhase.INITIATION)
                # Check if velocity is decreasing (completion)  
                elif i > 0 and smoothed_velocities[i] < smoothed_velocities[i-1]:
                    phases.append(MovementPhase.COMPLETION)
                else:
                    phases.append(MovementPhase.REST)
            else:
                phases.append(MovementPhase.EXECUTION)
                
        return phases
    
    def detect_body_movement_segments(self, min_duration=10) -> List[Dict]:
        """
        Detect body movements (WALK, SQUAT, LOWER_BACK_BEND) independently
        These are bilateral movements that should be the same for both sides
        
        Args:
            min_duration: minimum number of frames for a valid movement segment
        Returns:
            List of body movement segments with start/stop frames
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        try:
            # Use CENTER OF MASS (average of both hips) for velocity calculation
            left_hip_coords = self.get_joint_coordinates('hip', 'left')
            right_hip_coords = self.get_joint_coordinates('hip', 'right')
            center_of_mass = (left_hip_coords + right_hip_coords) / 2
            
            # Calculate velocity based on hip movement
            hip_velocities = self.calculate_joint_velocities(center_of_mass)
            
            # Also consider spine movement
            upper_spine_coords = self.get_joint_coordinates('upper_spine')
            spine_velocities = self.calculate_joint_velocities(upper_spine_coords)
            
            # Combined velocity (max of hip or spine movement)
            combined_velocities = np.maximum(hip_velocities, spine_velocities)
            
            # Smooth velocities to reduce noise
            from scipy import signal as sig
            window_size = min(21, len(combined_velocities) // 4)
            if window_size >= 3 and window_size % 2 == 0:
                window_size += 1
            if window_size >= 3:
                smoothed_velocities = sig.savgol_filter(combined_velocities, window_size, 2)
            else:
                smoothed_velocities = combined_velocities
            
            # Use fixed velocity threshold for body movements (lower than arm movements)
            velocity_threshold = 0.1  # m/s - body movements are typically slower
            
            # Find movement segments
            segments = []
            in_movement = False
            start_frame = 0
            
            for frame in range(len(smoothed_velocities)):
                current_velocity = smoothed_velocities[frame]
                
                if not in_movement and current_velocity > velocity_threshold:
                    # Movement start detected
                    start_frame = frame
                    in_movement = True
                    
                elif in_movement and current_velocity <= velocity_threshold:
                    # Movement end detected
                    duration = frame - start_frame
                    if duration >= min_duration:
                        
                        # Classify body movement for this segment
                        segment_data = self.data.iloc[start_frame:frame+1].copy()
                        
                        try:
                            movement_type = self._classify_body_movement(segment_data)
                            
                            # Only keep if it's actually a body movement
                            if movement_type in [MovementType.WALK, MovementType.SQUAT, 
                                                MovementType.LOWER_BACK_BEND, MovementType.TWIST,
                                                MovementType.ROTATE]:
                                
                                # Calculate segment metrics
                                hip_displacement = np.linalg.norm(center_of_mass[frame] - center_of_mass[start_frame])
                                max_velocity = np.max(smoothed_velocities[start_frame:frame+1])
                                
                                segments.append({
                                    'start_frame': start_frame,
                                    'end_frame': frame,
                                    'duration_frames': duration,
                                    'duration_seconds': duration / 60.0,
                                    'movement_type': movement_type.value,
                                    'wrist_displacement': hip_displacement,  # Using hip displacement for consistency
                                    'max_velocity': max_velocity,
                                    'avg_velocity': np.mean(smoothed_velocities[start_frame:frame+1]),
                                    'is_body_movement': True
                                })
                        except Exception as e:
                            print(f"Error classifying body movement: {e}")
                    
                    in_movement = False
            
            return segments
            
        except Exception as e:
            print(f"Error in body movement segmentation: {e}")
            return []
    
    def _classify_body_movement(self, segment_data: pd.DataFrame) -> MovementType:
        """
        Classify body movements only (not arm movements)
        Uses bilateral data (both sides averaged)
        
        Args:
            segment_data: DataFrame containing the segment to classify
        Returns:
            MovementType: WALK, SQUAT, LOWER_BACK_BEND, or UNKNOWN
        """
        try:
            # Get bilateral hip data (average of both sides)
            left_hip = segment_data[['left_hip_x', 'left_hip_y', 'left_hip_z']].values
            right_hip = segment_data[['right_hip_x', 'right_hip_y', 'right_hip_z']].values
            avg_hip = (left_hip + right_hip) / 2
            
            # Get bilateral knee and ankle data
            left_knee = segment_data[['left_knee_x', 'left_knee_y', 'left_knee_z']].values
            right_knee = segment_data[['right_knee_x', 'right_knee_y', 'right_knee_z']].values
            left_ankle = segment_data[['left_ankle_x', 'left_ankle_y', 'left_ankle_z']].values
            right_ankle = segment_data[['right_ankle_x', 'right_ankle_y', 'right_ankle_z']].values
            
            # Get spine data (already bilateral)
            upper_spine = segment_data[['spine_upper_x', 'spine_upper_y', 'spine_upper_z']].values
            
            # Get spine rotation data for twist/rotate detection
            spine_rotation = segment_data['upper_spine_rz'].values if 'upper_spine_rz' in segment_data.columns else np.array([])
            spine_rotation_range = np.max(spine_rotation) - np.min(spine_rotation) if len(spine_rotation) > 0 else 0
            
            # Get shoulder rotation for twist detection (differential rotation)
            left_shoulder_rot = segment_data['left_shoulder_rotation'].values if 'left_shoulder_rotation' in segment_data.columns else np.array([])
            right_shoulder_rot = segment_data['right_shoulder_rotation'].values if 'right_shoulder_rotation' in segment_data.columns else np.array([])
            shoulder_rotation_diff = np.mean(np.abs(left_shoulder_rot - right_shoulder_rot)) if len(left_shoulder_rot) > 0 and len(right_shoulder_rot) > 0 else 0
            
            # Get knee flexion angles for both legs
            left_knee_angles = segment_data['left_knee_flexion'].values if 'left_knee_flexion' in segment_data.columns else np.array([])
            right_knee_angles = segment_data['right_knee_flexion'].values if 'right_knee_flexion' in segment_data.columns else np.array([])
            
            # Calculate movement characteristics
            hip_horizontal_displacement = np.sqrt(
                (avg_hip[-1][0] - avg_hip[0][0])**2 +  # X
                (avg_hip[-1][1] - avg_hip[0][1])**2    # Y
            )
            hip_vertical_movement = avg_hip[-1][2] - avg_hip[0][2]  # Z
            
            spine_forward_movement = upper_spine[-1][1] - upper_spine[0][1]  # Y
            spine_vertical_movement = upper_spine[-1][2] - upper_spine[0][2]  # Z
            
            # Detect alternating leg movement
            alternating_steps = 0
            if len(left_knee_angles) > 10 and len(right_knee_angles) > 10:
                alternating_steps = self._detect_alternating_leg_movement(
                    left_knee_angles, right_knee_angles, left_ankle, right_ankle
                )
            
            left_knee_range = np.max(left_knee_angles) - np.min(left_knee_angles) if len(left_knee_angles) > 0 else 0
            right_knee_range = np.max(right_knee_angles) - np.min(right_knee_angles) if len(right_knee_angles) > 0 else 0
            
            # Calculate velocity (average of segment)
            hip_velocities = self.calculate_joint_velocities(avg_hip)
            max_velocity = np.max(hip_velocities) if len(hip_velocities) > 0 else 0
            
            # TWIST detection (torso rotation while standing - highest priority for rotation)
            # - Significant spine rotation
            # - Hips stay relatively stable (not walking/moving)
            # - Differential shoulder rotation (torso twist)
            # - Moderate velocity
            if (spine_rotation_range > MovementThresholds.Twist.MIN_SPINE_ROTATION and
                hip_horizontal_displacement < MovementThresholds.Twist.MAX_HIP_HORIZONTAL_DISPLACEMENT and
                shoulder_rotation_diff > MovementThresholds.Twist.MIN_SHOULDER_ROTATION_DIFF and
                max_velocity > MovementThresholds.Twist.MIN_VELOCITY):
                return MovementType.TWIST
            
            # ROTATE detection (full body rotation with movement)
            # - Significant spine rotation
            # - Body is moving horizontally
            # - Moderate velocity
            if (spine_rotation_range > MovementThresholds.Rotate.MIN_SPINE_ROTATION and
                hip_horizontal_displacement > MovementThresholds.Rotate.MIN_HIP_HORIZONTAL_DISPLACEMENT and
                max_velocity > MovementThresholds.Rotate.MIN_VELOCITY):
                return MovementType.ROTATE
            
            # WALK detection (highest priority for body movements)
            if (hip_horizontal_displacement > MovementThresholds.Walk.MIN_HORIZONTAL_DISPLACEMENT and
                alternating_steps >= MovementThresholds.Walk.MIN_ALTERNATING_STEPS and
                left_knee_range > MovementThresholds.Walk.MIN_KNEE_FLEXION_RANGE and
                right_knee_range > MovementThresholds.Walk.MIN_KNEE_FLEXION_RANGE and
                max_velocity > MovementThresholds.Walk.MIN_VELOCITY):
                return MovementType.WALK
            
            # SQUAT detection
            if (hip_vertical_movement < MovementThresholds.Squat.MIN_VERTICAL_HIP_MOVEMENT and
                max(left_knee_range, right_knee_range) > MovementThresholds.Squat.MIN_KNEE_BEND and
                abs(spine_forward_movement) < MovementThresholds.Squat.MAX_FORWARD_LEAN and
                max_velocity > MovementThresholds.Squat.MIN_VELOCITY):
                return MovementType.SQUAT
            
            # LOWER_BACK_BEND detection
            if (spine_forward_movement > MovementThresholds.LowerBackBend.MIN_FORWARD_SPINE_MOVEMENT and
                spine_vertical_movement < MovementThresholds.LowerBackBend.MIN_VERTICAL_DROP and
                hip_vertical_movement > MovementThresholds.LowerBackBend.MAX_HIP_VERTICAL_MOVEMENT and
                max_velocity > MovementThresholds.LowerBackBend.MIN_VELOCITY):
                return MovementType.LOWER_BACK_BEND
            
            return MovementType.UNKNOWN
            
        except Exception as e:
            print(f"Error in body movement classification: {e}")
            return MovementType.UNKNOWN
    
    def detect_movement_segments(self, side: str = 'left', min_duration=10) -> List[Dict]:
        """
        Detect start and stop frames for movement segments
        Combines arm movements (side-specific) and body movements (bilateral)
        
        Args:
            side: 'left' or 'right' to analyze
            min_duration: minimum number of frames for a valid movement segment
        Returns:
            List of movement segments with start/stop frames and movement type
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        try:
            # Step 1: Detect arm movements (side-specific)
            arm_segments = self._detect_arm_movement_segments(side, min_duration)
            
            # Step 2: Detect body movements (once, not per side)
            # Cache body segments to avoid re-computing for both sides
            if not hasattr(self, '_cached_body_segments'):
                self._cached_body_segments = self.detect_body_movement_segments(min_duration)
            
            body_segments = self._cached_body_segments
            
            # Step 3: Merge segments chronologically
            if body_segments is not None:
                all_segments = arm_segments + body_segments
            else:
                all_segments = arm_segments
            all_segments.sort(key=lambda x: x['start_frame'])
            
            return all_segments
            
        except Exception as e:
            print(f"Error in movement segmentation: {e}")
            return []
    
    def _detect_arm_movement_segments(self, side: str, min_duration: int) -> List[Dict]:
        """
        Detect arm movements only (REACH, PUSH, PULL, LIFT, CARRY)
        This is the original detect_movement_segments logic but filtered for arm movements
        
        Args:
            side: 'left' or 'right' to analyze
            min_duration: minimum number of frames for a valid movement segment
        Returns:
            List of arm movement segments
        """
        try:
            # Get wrist coordinates and calculate velocity
            wrist_coords = self.get_joint_coordinates('wrist', side)
            velocities = self.calculate_joint_velocities(wrist_coords)
            
            # Smooth velocities to reduce noise
            from scipy import signal as sig
            window_size = min(21, len(velocities) // 4)
            if window_size >= 3 and window_size % 2 == 0:
                window_size += 1
            if window_size >= 3:
                smoothed_velocities = sig.savgol_filter(velocities, window_size, 2)
            else:
                smoothed_velocities = velocities
            
            # Define movement threshold (adaptive based on data)
            velocity_threshold = np.mean(smoothed_velocities) + 0.5 * np.std(smoothed_velocities)
            
            # Find movement segments
            segments = []
            in_movement = False
            start_frame = 0
            
            for frame in range(len(smoothed_velocities)):
                current_velocity = smoothed_velocities[frame]
                
                if not in_movement and current_velocity > velocity_threshold:
                    # Movement start detected
                    start_frame = frame
                    in_movement = True
                    
                elif in_movement and current_velocity <= velocity_threshold:
                    # Movement end detected
                    duration = frame - start_frame
                    if duration >= min_duration:  # Only include significant movements
                        
                        # Classify the movement type for this segment
                        segment_data = self.data.iloc[start_frame:frame+1].copy()
                        original_data = self.data
                        self.data = segment_data
                        
                        try:
                            movement_type = self.classify_movement_type(side, arm_only=True)
                            
                            # Only keep arm movements (filter out body movements)
                            if movement_type.value in ['reach', 'push', 'pull', 'lift', 'carry', 'unknown']:
                                # Calculate segment metrics
                                wrist_displacement = np.linalg.norm(wrist_coords[frame] - wrist_coords[start_frame])
                                max_velocity = np.max(smoothed_velocities[start_frame:frame+1])
                                
                                segments.append({
                                    'start_frame': start_frame,
                                    'end_frame': frame,
                                    'duration_frames': duration,
                                    'duration_seconds': duration / 60.0,  # Assuming 60 FPS
                                    'movement_type': movement_type.value,
                                    'wrist_displacement': wrist_displacement,
                                    'max_velocity': max_velocity,
                                    'avg_velocity': np.mean(smoothed_velocities[start_frame:frame+1]),
                                    'is_body_movement': False
                                })
                        except Exception as e:
                            # If classification fails, mark as unknown
                            wrist_displacement = np.linalg.norm(wrist_coords[frame] - wrist_coords[start_frame])
                            max_velocity = np.max(smoothed_velocities[start_frame:frame+1])
                            segments.append({
                                'start_frame': start_frame,
                                'end_frame': frame,
                                'duration_frames': duration,
                                'duration_seconds': duration / 60.0,
                                'movement_type': 'unknown',
                                'wrist_displacement': wrist_displacement,
                                'max_velocity': max_velocity,
                                'avg_velocity': np.mean(smoothed_velocities[start_frame:frame+1]),
                                'is_body_movement': False
                            })
                        finally:
                            # Restore original data
                            self.data = original_data
                    
                    in_movement = False
            
            return segments
            
        except Exception as e:
            print(f"Error in arm movement segmentation: {e}")
            return []
    
    def classify_movement_type(self, side: str = 'left', arm_only: bool = False) -> MovementType:
        """
        Decision tree to classify movement type based on joint angles and trajectories
        Args:
            side: 'left' or 'right' to analyze
            arm_only: If True, only classify arm movements (skip body movements)
        Returns:
            Classified movement type
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        try:
            # Get upper body joint coordinates and calculate metrics
            wrist_coords = self.get_joint_coordinates('wrist', side)
            elbow_coords = self.get_joint_coordinates('elbow', side)
            shoulder_coords = self.get_joint_coordinates('shoulder', side)
            
            # Skip body movement detection if arm_only is True
            if not arm_only:
                # Get lower body and spine joint coordinates for squat and bend detection
                hip_coords = self.get_joint_coordinates('hip', side)
                knee_coords = self.get_joint_coordinates('knee', side)
                ankle_coords = self.get_joint_coordinates('ankle', side)
                upper_spine_coords = self.get_joint_coordinates('upper_spine')
                
                # Get opposite side leg for walk detection (alternating leg movement)
                opposite_side = 'right' if side == 'left' else 'left'
                opposite_knee_coords = self.get_joint_coordinates('knee', opposite_side)
                opposite_ankle_coords = self.get_joint_coordinates('ankle', opposite_side)
            else:
                # Set dummy values for arm-only classification
                hip_coords = None
                knee_coords = None
                ankle_coords = None
                upper_spine_coords = None
                opposite_knee_coords = None
                opposite_ankle_coords = None
            
            # Calculate joint angles
            angles = self.calculate_joint_angles(side)
            elbow_angles = angles.get(f'{side}_elbow_angle', np.array([]))
            shoulder_angles = angles.get(f'{side}_shoulder_angle', np.array([]))
            
            # Calculate knee angles for squat and walk detection (only if not arm_only)
            if not arm_only:
                opposite_side = 'right' if side == 'left' else 'left'
                knee_flexion_col = f'{side}_knee_flexion'
                opposite_knee_flexion_col = f'{opposite_side}_knee_flexion'
                if knee_flexion_col in self.data.columns:
                    knee_angles = self.data[knee_flexion_col].values
                else:
                    knee_angles = np.array([])
                
                if opposite_knee_flexion_col in self.data.columns:
                    opposite_knee_angles = self.data[opposite_knee_flexion_col].values
                else:
                    opposite_knee_angles = np.array([])
            else:
                knee_angles = np.array([])
                opposite_knee_angles = np.array([])
            
            if len(elbow_angles) == 0 or len(shoulder_angles) == 0:
                return MovementType.UNKNOWN
                
            # Calculate movement metrics
            wrist_velocities = self.calculate_joint_velocities(wrist_coords)
            
            # Decision tree logic
            return self._decision_tree_classify(
                wrist_coords, elbow_coords, shoulder_coords,
                elbow_angles, shoulder_angles, wrist_velocities,
                hip_coords, knee_coords, upper_spine_coords, knee_angles,
                ankle_coords, opposite_knee_coords, opposite_ankle_coords,
                opposite_knee_angles, arm_only
            )
            
        except Exception as e:
            print(f"Error in movement classification: {e}")
            return MovementType.UNKNOWN
    
    def _decision_tree_classify(self, 
                                    wrist_coords: np.ndarray, 
                                    elbow_coords: np.ndarray, 
                                    shoulder_coords: np.ndarray, 
                                    elbow_angles: np.ndarray,
                                    shoulder_angles: np.ndarray, 
                                    wrist_velocities: np.ndarray,
                                    hip_coords: np.ndarray,
                                    knee_coords: np.ndarray,
                                    upper_spine_coords: np.ndarray,
                                    knee_angles: np.ndarray,
                                    ankle_coords: np.ndarray,
                                    opposite_knee_coords: np.ndarray,
                                    opposite_ankle_coords: np.ndarray,
                                    opposite_knee_angles: np.ndarray,
                                    arm_only: bool = False) -> MovementType:
            """
            Core decision tree logic for movement classification
            """
            # Calculate upper body movement characteristics
            wrist_displacement = self._calculate_displacement(wrist_coords)
            elbow_angle_range = np.max(elbow_angles) - np.min(elbow_angles)
            shoulder_angle_range = np.max(shoulder_angles) - np.min(shoulder_angles)
            max_velocity = np.max(wrist_velocities)
            
            # Calculate trajectory characteristics for arm movements
            forward_movement = self._calculate_forward_movement(wrist_coords)
            vertical_movement = self._calculate_vertical_movement(wrist_coords)
            lateral_movement = self._calculate_lateral_movement(wrist_coords)
            
            # Calculate lower body and spine characteristics for squat and bend detection (skip if arm_only)
            if not arm_only and hip_coords is not None:
                hip_vertical_movement = self._calculate_vertical_movement(hip_coords)
                spine_forward_movement = self._calculate_forward_movement(upper_spine_coords)
                spine_vertical_movement = self._calculate_vertical_movement(upper_spine_coords)
            else:
                hip_vertical_movement = 0
                spine_forward_movement = 0
                spine_vertical_movement = 0
            
            # Calculate hip horizontal displacement and alternating steps for walk detection (skip if arm_only)
            if not arm_only and hip_coords is not None:
                hip_horizontal_displacement = np.sqrt(
                    self._calculate_forward_movement(hip_coords)**2 + 
                    self._calculate_lateral_movement(hip_coords)**2
                )
                
                # Calculate knee angle ranges for squat and walk detection
                if len(knee_angles) > 0:
                    knee_angle_range = np.max(knee_angles) - np.min(knee_angles)
                else:
                    knee_angle_range = 0
                    
                if len(opposite_knee_angles) > 0:
                    opposite_knee_angle_range = np.max(opposite_knee_angles) - np.min(opposite_knee_angles)
                else:
                    opposite_knee_angle_range = 0
                
                # Detect alternating leg movement for walking
                alternating_steps = self._detect_alternating_leg_movement(
                    knee_angles, opposite_knee_angles, ankle_coords, opposite_ankle_coords
                )
            else:
                hip_horizontal_displacement = 0
                knee_angle_range = 0
                opposite_knee_angle_range = 0
                alternating_steps = 0
            
            print(f"Movement Analysis:")
            print(f"  Wrist displacement: {wrist_displacement:.3f}m")
            print(f"  Elbow angle range: {elbow_angle_range:.1f}°")
            print(f"  Shoulder angle range: {shoulder_angle_range:.1f}°")
            print(f"  Max velocity: {max_velocity:.3f}m/s")
            print(f"  Forward movement: {forward_movement:.3f}m")
            print(f"  Vertical movement: {vertical_movement:.3f}m")
            print(f"  Lateral movement: {lateral_movement:.3f}m")
            if not arm_only:
                print(f"  Hip vertical movement: {hip_vertical_movement:.3f}m")
                print(f"  Spine forward movement: {spine_forward_movement:.3f}m")
                print(f"  Spine vertical movement: {spine_vertical_movement:.3f}m")
                print(f"  Knee angle range: {knee_angle_range:.1f}°")
                print(f"  Hip horizontal displacement: {hip_horizontal_displacement:.3f}m")
                print(f"  Alternating steps detected: {alternating_steps}")
            
            # Decision tree rules
            # Skip body movements if arm_only is True
            if not arm_only:
                # WALK movement criteria:
                # - Significant horizontal hip displacement
                # - Alternating leg movements (detected from knee flexion patterns)
                # - Both legs show knee flexion variation
                # - Moderate velocity
                if (hip_horizontal_displacement > MovementThresholds.Walk.MIN_HORIZONTAL_DISPLACEMENT and
                    alternating_steps >= MovementThresholds.Walk.MIN_ALTERNATING_STEPS and
                    knee_angle_range > MovementThresholds.Walk.MIN_KNEE_FLEXION_RANGE and
                    opposite_knee_angle_range > MovementThresholds.Walk.MIN_KNEE_FLEXION_RANGE and
                    max_velocity > MovementThresholds.Walk.MIN_VELOCITY):
                    return MovementType.WALK
                
                # SQUAT movement criteria:
                # - Significant downward hip movement (negative Z)
                # - Significant knee flexion
                # - Limited forward lean (relatively upright posture)
                # - Moderate velocity
                if (hip_vertical_movement < MovementThresholds.Squat.MIN_VERTICAL_HIP_MOVEMENT and
                    knee_angle_range > MovementThresholds.Squat.MIN_KNEE_BEND and
                    abs(spine_forward_movement) < MovementThresholds.Squat.MAX_FORWARD_LEAN and
                    max_velocity > MovementThresholds.Squat.MIN_VELOCITY):
                    return MovementType.SQUAT
                    
                # LOWER_BACK_BEND movement criteria:
                # - Significant forward spine movement
                # - Vertical drop of upper spine
                # - Hip stays relatively stable (not squatting)
                # - Moderate velocity
                elif (spine_forward_movement > MovementThresholds.LowerBackBend.MIN_FORWARD_SPINE_MOVEMENT and
                      spine_vertical_movement < MovementThresholds.LowerBackBend.MIN_VERTICAL_DROP and
                      hip_vertical_movement > MovementThresholds.LowerBackBend.MAX_HIP_VERTICAL_MOVEMENT and
                      max_velocity > MovementThresholds.LowerBackBend.MIN_VELOCITY):
                    return MovementType.LOWER_BACK_BEND
            
            # ARM MOVEMENTS (always check these)
            # REACH movement criteria:
            # - Significant wrist displacement (> threshold)
            # - Moderate elbow extension (angle range > threshold)
            # - Primary forward movement component
            # - Moderate peak velocity
            if (wrist_displacement > MovementThresholds.Reach.MIN_DISPLACEMENT and 
                elbow_angle_range > MovementThresholds.Reach.MIN_ELBOW_RANGE and
                forward_movement > max(vertical_movement, lateral_movement) and
                max_velocity > MovementThresholds.Reach.MIN_VELOCITY):
                return MovementType.REACH
                
            # PUSH movement criteria:
            # - Moderate to high velocity peak
            # - Any forward or lateral movement
            # - Elbow extension pattern
            # - Moderate displacement
            elif (max_velocity > MovementThresholds.Push.MIN_VELOCITY and
                  (forward_movement > MovementThresholds.Push.MIN_FORWARD_LATERAL or 
                   lateral_movement > MovementThresholds.Push.MIN_FORWARD_LATERAL) and
                  elbow_angle_range > MovementThresholds.Push.MIN_ELBOW_RANGE and
                  wrist_displacement > MovementThresholds.Push.MIN_DISPLACEMENT):
                return MovementType.PUSH
                
            # LIFT movement criteria:
            # - Primary vertical movement component
            # - Moderate displacement
            # - Shoulder flexion pattern
            elif (vertical_movement > max(forward_movement, lateral_movement) and
                  vertical_movement > MovementThresholds.Lift.MIN_VERTICAL_MOVEMENT and
                  shoulder_angle_range > MovementThresholds.Lift.MIN_SHOULDER_RANGE):
                return MovementType.LIFT
                
            # PULL movement criteria: 
            # - Negative forward movement (towards body)
            # - Elbow flexion pattern
            # - Moderate velocity
            elif (forward_movement < MovementThresholds.Pull.MIN_BACKWARD_MOVEMENT and
                  elbow_angle_range > MovementThresholds.Pull.MIN_ELBOW_RANGE and
                  max_velocity > MovementThresholds.Pull.MIN_VELOCITY):
                return MovementType.PULL
                
            # CARRY movement criteria:
            # - Sustained moderate velocity
            # - Limited angle changes (stable grip)
            # - Continuous lateral or forward movement
            elif (max_velocity > MovementThresholds.Carry.MIN_VELOCITY and
                  elbow_angle_range < MovementThresholds.Carry.MAX_ELBOW_RANGE and
                  shoulder_angle_range < MovementThresholds.Carry.MAX_SHOULDER_RANGE and
                  max(forward_movement, lateral_movement) > MovementThresholds.Carry.MIN_HORIZONTAL_MOVEMENT):
                return MovementType.CARRY
                
            else:
                return MovementType.UNKNOWN
    
    def _calculate_displacement(self, coords: np.ndarray) -> float:
        """Calculate total 3D displacement from start to end"""
        if len(coords) < 2:
            return 0.0
        start_pos = coords[0]
        end_pos = coords[-1]
        return np.linalg.norm(end_pos - start_pos)
    
    def _calculate_forward_movement(self, coords: np.ndarray) -> float:
        """Calculate net forward movement (Y-axis, positive away from body)"""
        if len(coords) < 2:
            return 0.0
        return coords[-1][1] - coords[0][1]
    
    def _calculate_vertical_movement(self, coords: np.ndarray) -> float:
        """Calculate net vertical movement (Z-axis, positive upward)"""
        if len(coords) < 2:
            return 0.0
        return coords[-1][2] - coords[0][2]
    
    def _calculate_lateral_movement(self, coords: np.ndarray) -> float:
        """Calculate net lateral movement (X-axis, positive to the right)"""
        if len(coords) < 2:
            return 0.0
        return abs(coords[-1][0] - coords[0][0])
    
    def _get_spine_rotation_range(self, start_frame: int, end_frame: int) -> float:
        """Calculate spine rotation angle range (rz component) during segment"""
        if self.data is None or 'upper_spine_rz' not in self.data.columns:
            return 0.0
        
        segment_data = self.data.iloc[start_frame:end_frame+1]
        spine_rz = segment_data['upper_spine_rz'].values
        
        if len(spine_rz) < 2:
            return 0.0
        
        # Return the range of rotation angles
        return np.max(spine_rz) - np.min(spine_rz)
    
    def _get_shoulder_rotation_diff(self, start_frame: int, end_frame: int, side: str) -> float:
        """Calculate difference in shoulder rotation between left and right shoulders"""
        if self.data is None:
            return 0.0
        
        segment_data = self.data.iloc[start_frame:end_frame+1]
        
        # Get shoulder rotation columns (if available)
        left_col = 'left_shoulder_rotation'
        right_col = 'right_shoulder_rotation'
        
        if left_col not in segment_data.columns or right_col not in segment_data.columns:
            return 0.0
        
        left_rotation = segment_data[left_col].values
        right_rotation = segment_data[right_col].values
        
        # Calculate average difference between shoulders
        rotation_diff = np.abs(left_rotation - right_rotation)
        return np.mean(rotation_diff)
    
    def _detect_alternating_leg_movement(self, 
                                         knee_angles: np.ndarray, 
                                         opposite_knee_angles: np.ndarray,
                                         ankle_coords: np.ndarray,
                                         opposite_ankle_coords: np.ndarray) -> int:
        """
        Detect alternating leg movements characteristic of walking
        Returns the number of alternating step cycles detected
        """
        if len(knee_angles) < 10 or len(opposite_knee_angles) < 10:
            return 0
        
        try:
            # Use scipy signal processing to find peaks in knee flexion
            from scipy.signal import find_peaks
            
            # Find peaks (maximum flexion) in both legs
            # Peaks indicate the swing phase of walking
            peaks_left, _ = find_peaks(knee_angles, height=np.mean(knee_angles), distance=5)
            peaks_right, _ = find_peaks(opposite_knee_angles, height=np.mean(opposite_knee_angles), distance=5)
            
            if len(peaks_left) == 0 or len(peaks_right) == 0:
                return 0
            
            # Count alternating patterns where legs move out of phase
            # In walking, when one leg is in swing phase (flexed), the other is in stance phase
            alternating_count = 0
            fps = 60  # Assuming 60 FPS
            
            # Check if peaks are alternating (not synchronized)
            all_peaks = sorted(list(peaks_left) + list(peaks_right))
            
            for i in range(len(all_peaks) - 1):
                time_diff = (all_peaks[i + 1] - all_peaks[i]) / fps
                
                # Check if time between peaks is within walking cadence range
                if (MovementThresholds.Walk.MIN_LEG_CYCLE_DURATION <= time_diff <= 
                    MovementThresholds.Walk.MAX_LEG_CYCLE_DURATION):
                    
                    # Check if consecutive peaks are from different legs (alternating)
                    peak1_is_left = all_peaks[i] in peaks_left
                    peak2_is_left = all_peaks[i + 1] in peaks_left
                    
                    if peak1_is_left != peak2_is_left:
                        alternating_count += 1
            
            return alternating_count
            
        except Exception as e:
            print(f"Error detecting alternating leg movement: {e}")
            return 0
    
    def analyze_movement(self, file_path: str, side: str = 'left') -> Dict:
        """
        Complete movement analysis pipeline
        Args:
            file_path: Path to the CSV file
            side: 'left' or 'right' side to analyze
        Returns:
            Dictionary with analysis results
        """
        # Load data
        data = self.load_data(file_path)
        if data is None:
            return {"error": "Failed to load data"}
        
        # Extract file name for movement type hint
        file_name = os.path.basename(file_path)
        print(f"\nAnalyzing movement from file: {file_name}")
        
        # Classify movement
        movement_type = self.classify_movement_type(side)
        
        # Calculate angles and trajectories
        angles = self.calculate_joint_angles(side)
        wrist_coords = self.get_joint_coordinates('wrist', side)
        wrist_velocities = self.calculate_joint_velocities(wrist_coords)
        
        # Detect movement phases
        movement_phases = self.detect_movement_phases(wrist_coords, wrist_velocities)
        
        # Calculate summary statistics
        results = {
            "file_name": file_name,
            "movement_type": movement_type,
            "side_analyzed": side,
            "n_frames": len(self.data),
            "duration_seconds": len(self.data) * (1/60),  # Assuming 60 FPS
            "angles": angles,
            "wrist_coordinates": wrist_coords,
            "wrist_velocities": wrist_velocities,
            "movement_phases": movement_phases,
            "max_velocity": np.max(wrist_velocities),
            "total_displacement": self._calculate_displacement(wrist_coords),
            "elbow_angle_range": np.max(angles.get(f'{side}_elbow_angle', [])) - np.min(angles.get(f'{side}_elbow_angle', [])) if angles.get(f'{side}_elbow_angle', []).size > 0 else 0,
            "shoulder_angle_range": np.max(angles.get(f'{side}_shoulder_angle', [])) - np.min(angles.get(f'{side}_shoulder_angle', [])) if angles.get(f'{side}_shoulder_angle', []).size > 0 else 0
        }
        
        return results
    
    def print_movement_summary(self, results: Dict):
        """Print a formatted summary of movement analysis"""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
            
        print(f"\n{'='*50}")
        print(f"MOVEMENT ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"File: {results['file_name']}")
        print(f"Classified Movement: {results['movement_type'].value.upper()}")
        print(f"Side Analyzed: {results['side_analyzed']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Frames: {results['n_frames']}")
        print(f"\nMovement Characteristics:")
        print(f"  Total Displacement: {results['total_displacement']:.3f}m") 
        print(f"  Maximum Velocity: {results['max_velocity']:.3f}m/s")
        print(f"  Elbow Angle Range: {results['elbow_angle_range']:.1f}°")
        print(f"  Shoulder Angle Range: {results['shoulder_angle_range']:.1f}°")
        
        # Count movement phases
        phase_counts = {}
        for phase in results['movement_phases']:
            phase_counts[phase.value] = phase_counts.get(phase.value, 0) + 1
            
        print(f"\nMovement Phases:")
        for phase, count in phase_counts.items():
            percentage = (count / results['n_frames']) * 100
            print(f"  {phase.title()}: {count} frames ({percentage:.1f}%)")
        print(f"{'='*50}")
    
    def analyze_all_files(self, folder_path: str = "Job Task Data", side: str = 'left') -> List[Dict]:
        """
        Analyze all CSV files in the specified folder
        Args:
            folder_path: Path to folder containing CSV files
            side: 'left' or 'right' side to analyze
        Returns:
            List of analysis results for each file
        """
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return []
            
        # Get all CSV files
        csv_files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))]
        
        if not csv_files:
            print(f"No CSV files found in {folder_path}")
            return []
            
        print(f"Found {len(csv_files)} CSV files to analyze:")
        for file in csv_files:
            print(f"  - {file}")
        
        results_list = []
        
        print(f"\n{'='*80}")
        print(f"BATCH ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        for i, csv_file in enumerate(csv_files, 1):
            file_path = os.path.join(folder_path, csv_file)
            print(f"\n[{i}/{len(csv_files)}] Analyzing: {csv_file}")
            print(f"-" * 60)
            
            try:
                results = self.analyze_movement(file_path, side)
                results_list.append(results)
                
                # Print compact summary
                if "error" not in results:
                    print(f"Movement Type: {results['movement_type'].value.upper()}")
                    print(f"Duration: {results['duration_seconds']:.1f}s | "
                          f"Displacement: {results['total_displacement']:.3f}m | "
                          f"Max Velocity: {results['max_velocity']:.3f}m/s")
                else:
                    print(f"ERROR: {results['error']}")
                    
            except Exception as e:
                print(f"ERROR analyzing {csv_file}: {e}")
                results_list.append({"file_name": csv_file, "error": str(e)})
        
        # Print summary statistics
        self._print_batch_summary(results_list)
        
        return results_list
    
    def _print_batch_summary(self, results_list: List[Dict]):
        """Print summary statistics for batch analysis"""
        
        print(f"\n{'='*80}")
        print(f"BATCH ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Count movement types
        movement_counts = {}
        successful_analyses = 0
        errors = 0
        
        for result in results_list:
            if "error" in result:
                errors += 1
            else:
                successful_analyses += 1
                movement_type = result['movement_type'].value
                movement_counts[movement_type] = movement_counts.get(movement_type, 0) + 1
        
        print(f"Total files processed: {len(results_list)}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Errors: {errors}")
        
        if movement_counts:
            print(f"\nMovement Type Distribution:")
            for movement_type, count in sorted(movement_counts.items()):
                percentage = (count / successful_analyses) * 100
                print(f"  {movement_type.upper()}: {count} files ({percentage:.1f}%)")
        
        # Calculate aggregate statistics for successful analyses
        if successful_analyses > 0:
            successful_results = [r for r in results_list if "error" not in r]
            
            total_displacements = [r['total_displacement'] for r in successful_results]
            max_velocities = [r['max_velocity'] for r in successful_results]
            durations = [r['duration_seconds'] for r in successful_results]
            
            print(f"\nAggregate Statistics:")
            print(f"  Average Displacement: {np.mean(total_displacements):.3f}m (±{np.std(total_displacements):.3f})")
            print(f"  Average Max Velocity: {np.mean(max_velocities):.3f}m/s (±{np.std(max_velocities):.3f})")
            print(f"  Average Duration: {np.mean(durations):.1f}s (±{np.std(durations):.1f})")
        
        print(f"{'='*80}")

    def analyze_dual_hand_industrial_movements(self, file_path: str, movement_types: List[str] = None) -> Dict:
        """
        Specialized analysis for industrial movements involving both hands
        Detects coordinated movements, synchronization, and industrial patterns
        """
        if movement_types is None:
            movement_types = ['push', 'pull', 'reach', 'lift', 'carry']
        
        # Analyze both hands
        left_results = self.analyze_movement(file_path, 'left')
        right_results = self.analyze_movement(file_path, 'right')
        
        # Extract movements
        left_movements = left_results.get('movements_found', {})
        right_movements = right_results.get('movements_found', {})
        
        # Find coordinated movements
        coordinated_movements = self._find_coordinated_movements(left_movements, right_movements)
        
        # Analyze industrial patterns
        industrial_patterns = self._analyze_industrial_patterns(left_movements, right_movements)
        
        return {
            'file_name': left_results.get('file_name', ''),
            'total_frames': left_results.get('total_frames', 0),
            'left_hand': left_movements,
            'right_hand': right_movements,
            'coordinated_movements': coordinated_movements,
            'industrial_patterns': industrial_patterns,
            'summary': self._create_dual_hand_summary(left_movements, right_movements, coordinated_movements)
        }
    
    def _find_coordinated_movements(self, left_movements: Dict, right_movements: Dict, sync_threshold: float = 1.0) -> List[Dict]:
        """Find movements that happen simultaneously between hands"""
        coordinated = []
        
        for left_type, left_segments in left_movements.items():
            for right_type, right_segments in right_movements.items():
                for left_seg in left_segments:
                    for right_seg in right_segments:
                        left_start, left_end = left_seg['start_time'], left_seg['end_time']
                        right_start, right_end = right_seg['start_time'], right_seg['end_time']
                        
                        # Calculate overlap
                        overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
                        time_gap = min(abs(left_start - right_end), abs(right_start - left_end))
                        
                        if overlap > 0 or time_gap <= sync_threshold:
                            coordinated.append({
                                'left_movement': left_type,
                                'right_movement': right_type,
                                'time_overlap': overlap,
                                'coordination_type': 'simultaneous' if overlap > 0 else 'sequential'
                            })
        
        return coordinated
    
    def _analyze_industrial_patterns(self, left_movements: Dict, right_movements: Dict) -> Dict:
        """Analyze patterns typical in industrial tasks"""
        patterns = {
            'bilateral_lifting': 0,
            'coordinated_pushing': 0,
            'bilateral_carry': 0,
            'hand_dominance': 'balanced'
        }
        
        # Determine hand dominance
        total_left = sum(len(segments) for segments in left_movements.values())
        total_right = sum(len(segments) for segments in right_movements.values())
        
        if total_left > total_right * 1.2:
            patterns['hand_dominance'] = 'left_dominant'
        elif total_right > total_left * 1.2:
            patterns['hand_dominance'] = 'right_dominant'
        
        return patterns
    
    def _create_dual_hand_summary(self, left_movements: Dict, right_movements: Dict, coordinated_movements: List) -> Dict:
        """Create comprehensive summary"""
        total_movements = sum(len(segments) for segments in left_movements.values()) + sum(len(segments) for segments in right_movements.values())
        
        return {
            'total_left_movements': sum(len(segments) for segments in left_movements.values()),
            'total_right_movements': sum(len(segments) for segments in right_movements.values()),
            'coordinated_actions': len(coordinated_movements),
            'coordination_ratio': len(coordinated_movements) / max(1, total_movements),
            'movement_distribution': {
                'left': {mv_type: len(segments) for mv_type, segments in left_movements.items()},
                'right': {mv_type: len(segments) for mv_type, segments in right_movements.items()}
            }
        }
    
    def print_industrial_movement_analysis(self, results: Dict):
        """Print comprehensive industrial movement analysis"""
        print(f"\nINDUSTRIAL DUAL-HAND MOVEMENT ANALYSIS")
        print(f"{'='*80}")
        print(f"File: {results['file_name']}")
        print(f"Total Frames: {results['total_frames']}")
        
        print(f"\nINDIVIDUAL HAND ACTIVITY:")
        print(f"  Left Hand: {results['summary']['total_left_movements']} movements")
        print(f"  Right Hand: {results['summary']['total_right_movements']} movements")
        
        print(f"\nCOORDINATION ANALYSIS:")
        print(f"  Coordinated Actions: {results['summary']['coordinated_actions']}")
        print(f"  Coordination Ratio: {results['summary']['coordination_ratio']:.2%}")
        
        patterns = results['industrial_patterns']
        print(f"\nINDUSTRIAL PATTERNS:")
        print(f"  Hand Dominance: {patterns['hand_dominance']}")
        
        if results['coordinated_movements']:
            print(f"\nCOORDINATED MOVEMENTS:")
            for i, coord in enumerate(results['coordinated_movements'][:3], 1):
                print(f"  {i}. {coord['left_movement'].upper()} + {coord['right_movement'].upper()} ({coord['coordination_type']})")
        
        print(f"\nMOVEMENT BREAKDOWN:")
        all_types = set(results['summary']['movement_distribution']['left'].keys()) | set(results['summary']['movement_distribution']['right'].keys())
        for mv_type in sorted(all_types):
            left_count = results['summary']['movement_distribution']['left'].get(mv_type, 0)
            right_count = results['summary']['movement_distribution']['right'].get(mv_type, 0)
            print(f"  {mv_type.upper()}: Left={left_count}, Right={right_count}, Total={left_count + right_count}")
        
        print(f"{'='*80}")


# Example usage and testing function
def test_movement_identification():
    """Test function to demonstrate the movement identification system"""
    
    # Initialize the movement identifier
    identifier = MovementIdentifier()
    
    # Test with the first CSV file (should be a REACH movement)
    file_path = "Job Task Data/2025-08-19-21-Reach---1.csv"
    
    if os.path.exists(file_path):
        print("Testing Movement Identification System")
        print("="*60)
        
        # Analyze movement
        results = identifier.analyze_movement(file_path, side='left')
        
        # Print summary
        identifier.print_movement_summary(results)
        
        return results
    else:
        print(f"Test file not found: {file_path}")
        return None


if __name__ == "__main__":
    # Run test when script is executed directly
    test_movement_identification()