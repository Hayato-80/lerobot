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

"""
Example: Real-time Disturbance Observer on SO101 Robot

This example demonstrates:
1. Connecting to a real SO101 robot
2. Reading motor current feedback
3. Computing disturbance estimates
4. Applying compensation in the control loop
5. Calibrating gravity torques with teleoperation
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calibrate_gravity_torques_with_teleoperation(robot, teleop, observer, duration_sec=60.0):
    """
    Calibrate gravity compensation while the user teleoperates the arm.

    This function:
    1. Lets the user move the robot through representative poses with the leader arm
    2. Collects current feedback during brief stationary pauses
    3. Estimates gravity torques from the collected samples
    """
    logger.info("Starting teleoperation-assisted gravity calibration for SO101...")
    logger.info("Move the leader arm through the range of motion and pause briefly at representative poses.")

    gravity_estimates = {joint: [] for joint in observer.joint_names}
    previous_positions = {}
    start_time = time.perf_counter()

    while time.perf_counter() - start_time < duration_sec:
        teleop_action = teleop.get_action()
        robot.send_action(teleop_action)
        obs = robot.get_observation()

        for joint in observer.joint_names:
            pos_key = f"{joint}.pos"
            current_key = f"{joint}.current"
            load_key = f"{joint}.load"

            if pos_key not in obs:
                continue

            pos = float(obs[pos_key])
            is_stationary = joint in previous_positions and abs(pos - previous_positions[joint]) < 0.25
            previous_positions[joint] = pos

            if not is_stationary:
                continue

            if current_key in obs:
                current_ma = float(obs[current_key])
                tau_gravity = current_ma * observer.current_to_torque_scale
            elif load_key in obs:
                tau_gravity = float(obs[load_key])
            else:
                continue

            gravity_estimates[joint].append(tau_gravity)

        time.sleep(0.01)

    # Average and set gravity torques from teleop-assisted samples
    for joint, estimates in gravity_estimates.items():
        if estimates:
            avg_gravity = float(np.median(estimates))
            observer.gravity_torque[joint] = avg_gravity
            logger.info(
                f"  {joint:20s}: median={avg_gravity:+.4f} Nm, "
                f"min={np.min(estimates):+.4f}, max={np.max(estimates):+.4f}, "
                f"std={np.std(estimates):.4f} from {len(estimates)} samples"
            )
        else:
            logger.warning(f"  {joint}: no stationary teleop samples collected")

    logger.info("Teleoperation-assisted gravity calibration complete!")


def _device_id_from_port(prefix: str, port: str) -> str:
    """Create a stable calibration id from a device port."""
    sanitized_port = port.replace("/", "_").replace(".", "_").replace("-", "_")
    return f"{prefix}_{sanitized_port}"


def monitor_disturbances(robot, observer, duration_sec=30, threshold_nm=3.0):
    """
    Monitor disturbances for a specified duration.

    Useful for:
    - Verifying observer is working
    - Detecting unexpected loads/collisions
    - Logging disturbance statistics
    """
    logger.info(f"Monitoring disturbances for {duration_sec} seconds (threshold={threshold_nm} Nm)...")

    disturbance_history = {joint: [] for joint in observer.joint_names}
    start_time = time.time()

    try:
        while time.time() - start_time < duration_sec:
            # Get observation with disturbance estimates
            obs = robot.get_observation()

            # Extract disturbances
            for joint in observer.joint_names:
                dist_key = f"{joint}.disturbance"
                if dist_key in obs:
                    dist = obs[dist_key]
                    disturbance_history[joint].append(dist)

                    # Alert if exceeds threshold
                    if abs(dist) > threshold_nm:
                        logger.warning(
                            f"  ⚠️  {joint}: {dist:+.3f} Nm (exceeds {threshold_nm} Nm threshold)"
                        )

            time.sleep(0.01)  # 100 Hz monitoring

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")

    # Print statistics
    logger.info("Disturbance Statistics:")
    for joint, history in disturbance_history.items():
        if history:
            history = np.array(history)
            logger.info(
                f"  {joint:20s}: "
                f"min={np.min(history):+.3f}, "
                f"mean={np.mean(history):+.3f}, "
                f"max={np.max(history):+.3f}, "
                f"std={np.std(history):.3f} Nm"
            )


def teach_with_disturbance_feedback(robot, teleop, observer, record_duration_sec=30):
    """
    Record a teleoperated trajectory with disturbance feedback.

    This demonstrates how to use disturbance estimates during teaching.
    Requires teleoperator input.
    """
    logger.info(f"Recording trajectory for {record_duration_sec} seconds...")
    logger.info("Move the leader arm to teleoperate the robot. Disturbances will be logged.")

    trajectory = []
    start_time = time.time()

    try:
        while time.time() - start_time < record_duration_sec:
            teleop_action = teleop.get_action()
            robot.send_action(teleop_action)
            obs = robot.get_observation()

            # Extract joint positions and disturbances
            frame = {
                "timestamp": time.time() - start_time,
                "positions": {},
                "disturbances": {},
            }

            for joint in observer.joint_names:
                pos_key = f"{joint}.pos"
                dist_key = f"{joint}.disturbance"

                if pos_key in obs:
                    frame["positions"][joint] = obs[pos_key]
                if dist_key in obs:
                    frame["disturbances"][joint] = obs[dist_key]

            trajectory.append(frame)
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Recording interrupted")

    logger.info(f"Recorded {len(trajectory)} frames")

    # Save trajectory
    output_path = Path("disturbance_trajectory.npz")
    trajectory_array = {
        "timestamps": np.array([f["timestamp"] for f in trajectory]),
        "positions": np.array([f["positions"] for f in trajectory]),
        "disturbances": np.array([f["disturbances"] for f in trajectory]),
    }
    np.savez(output_path, **trajectory_array)
    logger.info(f"Saved trajectory to {output_path}")

    return trajectory


def control_loop_with_compensation(robot, observer, target_actions, compensate=True):
    """
    Execute target actions with optional disturbance compensation.

    Args:
        robot: SOFollowerWithDisturbanceObserver instance
        observer: DisturbanceObserverStep instance
        target_actions: List of action dicts
        compensate: Whether to apply disturbance compensation
    """
    logger.info(
        f"Executing {len(target_actions)} actions "
        f"({'with' if compensate else 'without'} compensation)..."
    )

    for i, action in enumerate(target_actions):
        # Log progress every 100 actions to avoid spam for large trajectories
        if i % 100 == 0:
            logger.info(f"  Action {i+1}/{len(target_actions)}")

        # Execute with or without compensation
        robot.send_action(
            action,
            apply_compensation=compensate,
            compensation_scale=0.8,  # 80% of estimated disturbance
        )

        # Send commands at ~100 Hz without waiting for each to reach target.
        # The robot's servo controllers handle trajectory following internally.
        time.sleep(0.01)

    logger.info("Control loop complete")


def main():
    """Main example runner."""
    parser = argparse.ArgumentParser(description="SO Follower Disturbance Observer Example")
    parser.add_argument(
        "--port", type=str, default="/dev/ttyUSB0", help="Serial port for SO101 robot"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["calibrate", "monitor", "teach", "control"],
        default="monitor",
        help="Operation mode: "
        "'calibrate'=calibrate gravity with teleop, "
        "'monitor'=watch for disturbances (no teleop), "
        "'teach'=record trajectory with teleop and disturbance feedback, "
        "'control'=execute predefined actions with compensation",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Duration for monitoring (seconds)"
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0, help="Disturbance threshold for alerts (Nm)"
    )
    parser.add_argument(
        "--teleop-port",
        type=str,
        default=None,
        help="Serial port for the SO leader teleoperator used during calibration",
    )
    parser.add_argument(
        "--calibration-duration",
        type=float,
        default=30.0,
        help="Duration for teleoperation-assisted calibration (seconds)",
    )
    parser.add_argument(
        "--follower-id",
        type=str,
        default=None,
        help="Calibration ID/name for the follower robot (e.g., 'white'). If not provided, uses port-based ID.",
    )
    parser.add_argument(
        "--leader-id",
        type=str,
        default=None,
        help="Calibration ID/name for the leader teleoperator (e.g., 'black'). If not provided, uses port-based ID.",
    )
    parser.add_argument(
        "--skip-calibration-prompt",
        action="store_true",
        help="Skip motor calibration prompt and use existing calibration file (useful for pre-calibrated configs)",
    )
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        default=Path("disturbance_trajectory.npz"),
        help="Path to a recorded trajectory .npz produced by teach mode (optional)",
    )
    parser.add_argument(
        "--no-compensation",
        action="store_true",
        help="Disable disturbance compensation during control replay",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging to see current measurements during calibration",
    )

    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("lerobot").setLevel(logging.DEBUG)

    # Import robot and observer classes
    try:
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower_disturbance_observer import (
            SOFollowerWithDisturbanceObserver,
        )
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        from lerobot.processor.disturbance_observer import DisturbanceObserverStep
    except ImportError as e:
        logger.error(f"Failed to import LeRobot components: {e}")
        return

    # Create robot and observer
    logger.info(f"Connecting to SO101 on {args.port}...")
    follower_id = args.follower_id if args.follower_id else _device_id_from_port("so101_follower", args.port)
    config = SO101FollowerConfig(port=args.port, id=follower_id)
    robot = SOFollowerWithDisturbanceObserver(config)

    try:
        if args.mode == "calibrate" and args.teleop_port is None:
            parser.error("--teleop-port is required for teleoperation-assisted calibration")

        # Connect with or without calibration prompt based on flag
        should_calibrate = not args.skip_calibration_prompt
        robot.connect(calibrate=should_calibrate)
        logger.info("Robot connected!")

        # Create disturbance observer with custom parameters
        observer = DisturbanceObserverStep(
            current_to_torque_scale=0.00338*6.5,  # Nm per mA for STS3215
            gravity_torque={
                # For calibration, these will be updated based on teleop-assisted estimates. For monitoring/control, these can be set from prior calibration or left at 0.
                # "shoulder_pan": 0.0,
                # "shoulder_lift": 2.5,  # Large gravity load
                # "elbow_flex": 1.8,
                # "wrist_flex": 0.2,
                # "wrist_roll": 0.0,
                # "gripper": 0.0,

                # After calibration
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0220,
                "elbow_flex": 0.0439,
                "wrist_flex": 0.0220,
                "wrist_roll": 0.0220,
                "gripper": 0.00,
            },
            friction_coefficient={
                "shoulder_pan": 0.01,
                "shoulder_lift": 0.01,
                "elbow_flex": 0.01,
                "wrist_flex": 0.005,
                "wrist_roll": 0.005,
                "gripper": 0.001,
            },
            dt=0.01,
            velocity_filter_window=5,
        )

        robot.set_disturbance_observer(observer)

        # Execute requested mode
        if args.mode == "calibrate":
            leader_id = args.leader_id if args.leader_id else _device_id_from_port("so101_leader", args.teleop_port)
            teleop = SO101Leader(
                SO101LeaderConfig(
                    port=args.teleop_port,
                    id=leader_id,
                )
            )
            try:
                teleop.connect()
                calibrate_gravity_torques_with_teleoperation(
                    robot,
                    teleop,
                    observer,
                    duration_sec=args.calibration_duration,
                )
            finally:
                teleop.disconnect()

        elif args.mode == "monitor":
            monitor_disturbances(
                robot,
                observer,
                duration_sec=args.duration,
                threshold_nm=args.threshold,
            )

        elif args.mode == "teach":
            if args.teleop_port is None:
                logger.error("--teleop-port is required for teach mode")
                return
            leader_id = args.leader_id if args.leader_id else _device_id_from_port("so101_leader", args.teleop_port)
            teleop = SO101Leader(
                SO101LeaderConfig(
                    port=args.teleop_port,
                    id=leader_id,
                )
            )
            try:
                teleop.connect()
                teach_with_disturbance_feedback(
                    robot,
                    teleop,
                    observer,
                    record_duration_sec=30,
                )
            finally:
                teleop.disconnect()

        elif args.mode == "control":
            # If a recorded trajectory with `actions` exists, replay it; otherwise use a small demo
            if args.trajectory_path.exists():
                data = np.load(args.trajectory_path, allow_pickle=True)
                if "actions" in data:
                    actions = data["actions"]
                    # If joint names provided, build per-row action dicts
                    if "joint_names" in data:
                        joint_names = [str(j) for j in data["joint_names"]]
                        target_actions = []
                        for row in actions:
                            action = {}
                            for jname, val in zip(joint_names, row):
                                if not np.isnan(val):
                                    action[f"{jname}.pos"] = float(val)
                            if action:
                                target_actions.append(action)
                    else:
                        # actions might already be an array of dict-like objects
                        target_actions = [dict(a) for a in actions]

                elif "positions" in data:
                    # Fallback: replay recorded joint positions (encoder units) if actions missing
                    positions = data["positions"]
                    target_actions = []
                    
                    # positions is 1D array of dicts like {'joint_name': value, ...}
                    if positions.ndim == 1 and positions.dtype == object:
                        for pos_dict in positions:
                            if isinstance(pos_dict, dict):
                                # Convert dict keys to full action keys (e.g., 'shoulder_pan' → 'shoulder_pan.pos')
                                action = {f"{k}.pos": float(v) for k, v in pos_dict.items() if not np.isnan(v)}
                                if action:
                                    target_actions.append(action)
                            else:
                                logger.warning(f"Unexpected positions element type: {type(pos_dict)}")
                    else:
                        logger.error(f"Unexpected positions shape/dtype: ndim={positions.ndim}, dtype={positions.dtype}")
                        return

                    if not target_actions:
                        logger.error(f"No valid replay actions found in {args.trajectory_path}")
                        return
                else:
                    logger.warning(
                        f"No supported trajectory field found in {args.trajectory_path}; running small demo instead"
                    )
                    target_actions = [
                        {"shoulder_pan.pos": -0.1, "shoulder_lift.pos": 0.5, "elbow_flex.pos": 0.3, "wrist_flex.pos": 0.2, "wrist_roll.pos": 0.1, "gripper.pos": 0.0},
                        {"shoulder_pan.pos": 0.1, "shoulder_lift.pos": -0.3, "elbow_flex.pos": 0.0, "wrist_flex.pos": -0.2, "wrist_roll.pos": -0.1, "gripper.pos": 0.5},
                        {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0, "wrist_roll.pos": 0.0, "gripper.pos": 0.0},
                    ]
            else:
                # Demo fallback actions if no trajectory file found
                target_actions = [
                    {"shoulder_pan.pos": -0.1, "shoulder_lift.pos": 0.5, "elbow_flex.pos": 0.3, "wrist_flex.pos": 0.2, "wrist_roll.pos": 0.1, "gripper.pos": 0.0},
                    {"shoulder_pan.pos": 0.1, "shoulder_lift.pos": -0.3, "elbow_flex.pos": 0.0, "wrist_flex.pos": -0.2, "wrist_roll.pos": -0.1, "gripper.pos": 0.5},
                    {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0, "wrist_roll.pos": 0.0, "gripper.pos": 0.0},
                ]
                logger.info("No trajectory file found; running demo actions")

            
            
            control_loop_with_compensation(
                robot, observer, target_actions, compensate=not args.no_compensation
            )

    except Exception as e:
        logger.error(f"Error during operation: {e}", exc_info=True)

    finally:
        logger.info("Disconnecting robot...")
        try:
            robot.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

        logger.info("Done!")


if __name__ == "__main__":
    main()
