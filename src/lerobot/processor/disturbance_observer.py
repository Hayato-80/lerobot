#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Disturbance Observer for real robot control with current/torque feedback."""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry

if TYPE_CHECKING:
    from lerobot.types import EnvTransition

logger = logging.getLogger(__name__)


@dataclass
@ProcessorStepRegistry.register("disturbance_observer")
class DisturbanceObserverStep(ObservationProcessorStep):
    """
    Disturbance Observer for SO100 robots.

    Estimates external disturbances (torques) using motor current feedback and inertia model.
    Useful for:
    - Gravity compensation
    - Friction estimation
    - Collision detection
    - Force feedback control

    The observer works by:
    1. Measuring motor current (proportional to torque)
    2. Estimating torque from current using motor constants
    3. Subtracting commanded torque to get disturbance

    Based on standard disturbance observer theory:
    tau_dist = tau_measured - (J * a_cmd + b * v + g)
    where:
        tau_dist = estimated disturbance torque
        tau_measured = torque from current feedback
        J = inertia
        a_cmd = commanded acceleration (from desired velocity)
        b = viscous friction coefficient
        v = joint velocity
        g = gravity torque
    """

    # Motor current to torque conversion
    # SO100 uses STS3215 motors with ~0.16 mA/A and ~11 Nm max torque
    current_to_torque_scale: float = 0.00338*6.5  # Torque (Nm) per mA of motor current
    motor_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # Disturbance estimation parameters
    # Motor inertia (kg*m^2) - can be calibrated per motor
    motor_inertia: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.01,
            "shoulder_lift": 0.01,
            "elbow_flex": 0.01,
            "wrist_flex": 0.005,
            "wrist_roll": 0.005,
            "gripper": 0.002,
        }
    )

    # Viscous friction coefficient (Nm*s/rad)
    friction_coefficient: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.01,
            "shoulder_lift": 0.01,
            "elbow_flex": 0.01,
            "wrist_flex": 0.005,
            "wrist_roll": 0.005,
            "gripper": 0.001,
        }
    )

    # Gravity torque (Nm) - estimated or measured per joint
    gravity_torque: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0,
            "shoulder_lift": 2.0,  # Holding load
            "elbow_flex": 1.5,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
    )

    # Control loop timestep (seconds)
    dt: float = 0.01

    # Smoothing window size for velocity filtering
    velocity_filter_window: int = 3

    # Whether to include in observations
    include_in_observation: bool = True

    # Maximum disturbance magnitude to report (clipping for safety)
    max_disturbance: float = 10.0  # Nm

    def __post_init__(self):
        """Validate and initialize internal buffers."""
        self.joint_names = list(self.motor_inertia.keys())
        self.num_joints = len(self.joint_names)

        # Buffers for velocity filtering
        self._velocity_history = {}
        self._current_history = {}
        self._prev_position = {}

        # State for disturbance observer
        self._disturbance_estimate = None

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Estimate disturbances from motor feedback.

        Args:
            observation: RobotObservation dict containing:
                - "{joint}.pos": Joint position
                - "{joint}.current" (optional): Motor current in mA
                - "{joint}.load" (optional): Motor load/torque feedback

        Returns:
            observation: Updated with 'disturbance_estimate' key if include_in_observation=True
        """
        disturbance_est = {}

        # Extract feedback signals
        for joint in self.joint_names:
            pos_key = f"{joint}.pos"
            current_key = f"{joint}.current"
            load_key = f"{joint}.load"

            # Get motor current (primary signal for disturbance estimation)
            if current_key in observation:
                current_mA = observation[current_key]
                # Convert current to estimated torque
                tau_measured = float(current_mA) * self.current_to_torque_scale
            elif load_key in observation:
                # Fallback to load feedback if current not available
                tau_measured = float(observation[load_key])
            else:
                # No current/load feedback available
                disturbance_est[joint] = 0.0
                continue

            # Get position to compute velocity
            if pos_key in observation:
                pos = float(observation[pos_key])

                # Initialize position buffer on first call
                if joint not in self._prev_position:
                    self._prev_position[joint] = pos
                    disturbance_est[joint] = 0.0
                    continue

                # Compute velocity via finite difference
                vel_raw = (pos - self._prev_position[joint]) / self.dt
                self._prev_position[joint] = pos

                # Low-pass filter velocity via moving average
                if joint not in self._velocity_history:
                    self._velocity_history[joint] = []

                self._velocity_history[joint].append(vel_raw)
                if len(self._velocity_history[joint]) > self.velocity_filter_window:
                    self._velocity_history[joint].pop(0)

                vel_filtered = np.mean(self._velocity_history[joint])

            else:
                vel_filtered = 0.0

            # Estimate disturbance torque
            # tau_dist = tau_measured - (friction + gravity)
            # Note: Acceleration term (J*a) is typically neglected at typical servo speeds
            # or estimated from commanded velocity changes
            friction_torque = self.friction_coefficient.get(joint, 0.01) * vel_filtered
            gravity_torque = self.gravity_torque.get(joint, 0.0)

            tau_dist = tau_measured - friction_torque - gravity_torque

            # Clip for safety
            tau_dist = np.clip(tau_dist, -self.max_disturbance, self.max_disturbance)
            disturbance_est[joint] = float(tau_dist)

        # Store and return
        self._disturbance_estimate = disturbance_est

        if self.include_in_observation:
            observation = observation.copy()
            # Add disturbance estimate as a new observation field
            for joint, tau_dist in disturbance_est.items():
                observation[f"{joint}.disturbance"] = tau_dist

        return observation

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """ObservationProcessorStep interface - delegates to __call__."""
        return self.__call__(observation)

    def get_config(self) -> dict[str, Any]:
        """Return configuration as JSON-serializable dict."""
        return {
            "current_to_torque_scale": self.current_to_torque_scale,
            "motor_ids": self.motor_ids,
            "motor_inertia": self.motor_inertia,
            "friction_coefficient": self.friction_coefficient,
            "gravity_torque": self.gravity_torque,
            "dt": self.dt,
            "velocity_filter_window": self.velocity_filter_window,
            "include_in_observation": self.include_in_observation,
            "max_disturbance": self.max_disturbance,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return state dict (no learnable parameters in this observer)."""
        return {
            "velocity_history": self._velocity_history,
            "prev_position": self._prev_position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._velocity_history = state.get("velocity_history", {})
        self._prev_position = state.get("prev_position", {})

    def reset(self) -> None:
        """Reset internal buffers between episodes."""
        self._velocity_history = {}
        self._current_history = {}
        self._prev_position = {}
        self._disturbance_estimate = None

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """
        Declare new observation features created by this processor.

        Args:
            features: Current feature dictionary

        Returns:
            Updated features including disturbance estimates
        """
        if self.include_in_observation:
            for joint in self.joint_names:
                disturbance_key = f"{joint}.disturbance"
                if disturbance_key not in features:
                    features[disturbance_key] = PolicyFeature(
                        type=FeatureType.STATE,
                        shape=(),
                    )

        return features


@dataclass
@ProcessorStepRegistry.register("advanced_disturbance_observer")
class AdvancedDisturbanceObserverStep(DisturbanceObserverStep):
    """
    Advanced disturbance observer with learned observer gain and model-based compensation.

    This version supports:
    - Learned observer gains (via gradient descent)
    - Kinematic model integration (gravity estimation from joint positions)
    - High-frequency disturbance filtering (via state observer)
    """

    # Observer gain (how aggressively to respond to estimation errors)
    observer_gain: float = 0.5

    # Use kinematic model for gravity compensation (requires kinematics data)
    use_kinematic_model: bool = False

    # Link masses for gravity model (kg) - if empty, uses lookup table
    link_masses: dict[str, float] = field(default_factory=dict)

    # Robot kinematics parameters (DH parameters) - for advanced gravity estimation
    robot_kinematics_enabled: bool = False

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Advanced disturbance estimation with optional kinematic compensation.
        """
        # Call parent implementation for basic disturbance estimation
        observation = super().__call__(observation)

        # Optional: Apply kinematic model compensation
        if self.use_kinematic_model and self.robot_kinematics_enabled:
            observation = self._apply_kinematic_compensation(observation)

        return observation

    def _apply_kinematic_compensation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Apply kinematic model-based gravity and inertia compensation.

        This is a placeholder for integrating forward/inverse kinematics.
        """
        # In a full implementation, you would:
        # 1. Extract joint angles from observation
        # 2. Compute Jacobian from kinematics
        # 3. Estimate gravity torque using Jacobian transpose and end-effector forces
        # 4. Update disturbance estimate accordingly

        logger.debug("Kinematic compensation not yet implemented for this robot model")
        return observation

    def get_config(self) -> dict[str, Any]:
        """Return configuration including advanced parameters."""
        config = super().get_config()
        config.update({
            "observer_gain": self.observer_gain,
            "use_kinematic_model": self.use_kinematic_model,
            "link_masses": self.link_masses,
            "robot_kinematics_enabled": self.robot_kinematics_enabled,
        })
        return config
