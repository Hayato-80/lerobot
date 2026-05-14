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

"""Enhanced SO Follower (SO100/SO101) robot class with disturbance observer support.

This extends the standard SOFollower to:
1. Read additional motor feedback (current, load)
2. Compute disturbance estimates in real-time
3. Apply compensation in the control loop (optional)
"""

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from lerobot.robots.so_follower import SO100Follower, SOFollowerRobotConfig
from lerobot.types import RobotAction, RobotObservation

if TYPE_CHECKING:
    from lerobot.processor.disturbance_observer import DisturbanceObserverStep

logger = logging.getLogger(__name__)


class SOFollowerWithDisturbanceObserver(SO100Follower):
    """SO Follower robot (SO100/SO101) with real-time disturbance observation.

    This class extends SOFollower to:
    - Read Present_Current and Present_Load from motors
    - Estimate external disturbances (torques) on each joint
    - Optionally apply compensation in the control loop

    Example:
        from lerobot.robots.so_follower import SO100FollowerConfig
        from lerobot.robots.so_follower import SOFollowerWithDisturbanceObserver
        from lerobot.processor.disturbance_observer import DisturbanceObserverStep

        config = SO100FollowerConfig(port="/dev/ttyUSB0")
        robot = SOFollowerWithDisturbanceObserver(config)
        robot.connect()

        observer = DisturbanceObserverStep(gravity_torque={"shoulder_lift": 2.5})
        robot.set_disturbance_observer(observer)

        obs = robot.get_observation()
        print(obs["shoulder_lift.disturbance"])  # Estimated disturbance torque in Nm

        robot.disconnect()
    """

    def __init__(self, config: SOFollowerRobotConfig):
        """Initialize robot with disturbance observer support."""
        super().__init__(config)
        self._init_disturbance_observer()

    def _init_disturbance_observer(self):
        """Initialize disturbance observer parameters."""
        from lerobot.processor.disturbance_observer import DisturbanceObserverStep

        self.disturbance_observer: DisturbanceObserverStep | None = None
        self._prev_position = {}
        self._disturbance_estimates = {}

        # Auto-enable if motor bus supports current reading
        if hasattr(self.bus, "can_read_current"):
            logger.info("Motor bus supports current reading - disturbance observer enabled")

    def set_disturbance_observer(self, observer: "DisturbanceObserverStep") -> None:
        """Set custom disturbance observer."""
        self.disturbance_observer = observer

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """Extended features including disturbance estimates."""
        features = super().observation_features

        # Add disturbance estimate features if observer is configured
        if self.disturbance_observer:
            for motor in self.bus.motors:
                features[f"{motor}.disturbance"] = float

        return features

    def get_observation(self) -> RobotObservation:
        """Get observation with disturbance estimates.

        Returns:
            RobotObservation dict containing:
            - Standard features: position, images
            - New features: current, load, disturbance (if available)
        """
        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Try to read additional feedback (current, load) for disturbance estimation
        try:
            start = time.perf_counter()
            current_dict = self.bus.sync_read("Present_Current")
            obs_dict.update({f"{motor}.current": val for motor, val in current_dict.items()})
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read current: {dt_ms:.1f}ms")
        except Exception as e:
            logger.debug(f"Could not read Present_Current: {e}")

        try:
            start = time.perf_counter()
            load_dict = self.bus.sync_read("Present_Load")
            obs_dict.update({f"{motor}.load": val for motor, val in load_dict.items()})
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read load: {dt_ms:.1f}ms")
        except Exception as e:
            logger.debug(f"Could not read Present_Load: {e}")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        # Compute disturbance estimates if observer is configured
        if self.disturbance_observer:
            obs_dict = self.disturbance_observer.observation(obs_dict)
            # Store for later use (e.g., in action compensation)
            if hasattr(self.disturbance_observer, "_disturbance_estimate"):
                self._disturbance_estimates = self.disturbance_observer._disturbance_estimate

        return obs_dict

    def send_action(
        self,
        action: RobotAction,
        apply_compensation: bool = False,
        compensation_scale: float = 1.0,
    ) -> RobotAction:
        """Send action to motors with optional disturbance compensation.

        Args:
            action: Target joint positions (radians or degrees depending on config)
            apply_compensation: Whether to apply disturbance compensation to goal position
            compensation_scale: Scale factor for compensation (0.0 = no compensation, 1.0 = full)

        Returns:
            RobotAction: Actual action sent (after potential clipping/compensation)
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Optional: Apply disturbance compensation
        if apply_compensation and self._disturbance_estimates:
            logger.debug(f"Applying disturbance compensation: {self._disturbance_estimates}")
            for motor_name, disturbance_torque in self._disturbance_estimates.items():
                if motor_name in goal_pos:
                    # Add compensation term (negative of estimated disturbance)
                    compensation = -disturbance_torque * compensation_scale

                    # Convert torque compensation to position change
                    # Simplified: assume ~0.1 rad per Nm for typical SO Follower motors
                    position_compensation = compensation * 0.1

                    goal_pos[motor_name] = goal_pos[motor_name] + position_compensation

                    logger.debug(
                        f"  {motor_name}: disturbance={disturbance_torque:.3f} Nm, "
                        f"  compensation={position_compensation:.4f} rad"
                    )

        # Cap goal position when too far away from present position
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            from lerobot.robots.utils import ensure_safe_goal_position

            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}


# Convenience aliases for backward compatibility
SO100FollowerWithDisturbanceObserver = SOFollowerWithDisturbanceObserver
SO101FollowerWithDisturbanceObserver = SOFollowerWithDisturbanceObserver
