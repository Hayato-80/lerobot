#!/usr/bin/env python
"""
Simple helper to estimate Feetech `Present_Current` -> torque scaling.

Usage:
    uv run python examples/feetech_current_scale_test.py --port /dev/ttyACM0 --motor shoulder_lift --mass 0.1 --lever 0.05

Procedure:
- Run without any external load and press Enter to record baseline.
- Attach a known mass at a known lever arm (meters) to produce torque = m*g*l.
- Press Enter to record loaded current and compute implied Nm per current-unit.

This script prints raw "Present_Current" values and the inferred scale (Nm per current unit).
"""

import argparse
import time
import statistics
import math

from pathlib import Path


def sample_current(robot, motor_name: str, samples: int = 200, interval: float = 0.01):
    vals = []
    for _ in range(samples):
        obs = robot.get_observation()
        key = f"{motor_name}.current"
        if key in obs:
            vals.append(float(obs[key]))
        else:
            raise RuntimeError(f"Motor current key not present in observation: {key}")
        time.sleep(interval)
    return vals


def main():
    parser = argparse.ArgumentParser(description="Estimate Present_Current -> torque scale for Feetech motors")
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--motor", type=str, required=True, help="Motor name in robot config (e.g. shoulder_lift)")
    parser.add_argument("--mass", type=float, default=None, help="Known mass in kg (optional). If provided, script computes torque=m*g*l")
    parser.add_argument("--lever", type=float, default=None, help="Lever arm in meters (optional)")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--interval", type=float, default=0.01)

    args = parser.parse_args()

    try:
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower_disturbance_observer import (
            SOFollowerWithDisturbanceObserver,
        )
    except Exception as e:
        print(f"Failed to import robot classes: {e}")
        return

    config = SO101FollowerConfig(port=args.port)
    robot = SOFollowerWithDisturbanceObserver(config)

    try:
        print(f"Connecting to robot on {args.port}...")
        robot.connect()
        print("Connected. Make sure the robot is powered and torque enabled. Keep the arm stationary.")

        input("When ready (no external load attached), press Enter to record baseline current samples...")
        baseline = sample_current(robot, args.motor, samples=args.samples, interval=args.interval)
        base_med = statistics.median(baseline)
        base_mean = statistics.mean(baseline)
        base_std = statistics.stdev(baseline) if len(baseline) > 1 else 0.0
        print(f"Baseline ({len(baseline)} samples): median={base_med:.3f}, mean={base_mean:.3f}, std={base_std:.3f}")

        input("Now attach the known mass/lever (or apply your known torque) and keep arm stationary, then press Enter to record loaded samples...")
        loaded = sample_current(robot, args.motor, samples=args.samples, interval=args.interval)
        load_med = statistics.median(loaded)
        load_mean = statistics.mean(loaded)
        load_std = statistics.stdev(loaded) if len(loaded) > 1 else 0.0
        print(f"Loaded ({len(loaded)} samples): median={load_med:.3f}, mean={load_mean:.3f}, std={load_std:.3f}")

        delta_med = load_med - base_med
        delta_mean = load_mean - base_mean
        print(f"Delta current (median): {delta_med:.3f} units; (mean): {delta_mean:.3f} units")

        # If user provided mass & lever, compute torque
        if args.mass is not None and args.lever is not None:
            g = 9.80665
            torque_nm = args.mass * g * args.lever
            print(f"Applied external torque (m={args.mass} kg, l={args.lever} m): {torque_nm:.6f} Nm")

            if abs(delta_med) < 1e-6:
                print("Measured current delta is too small to compute a reliable scale. Increase the load or check wiring.")
            else:
                inferred_scale_med = torque_nm / delta_med
                inferred_scale_mean = torque_nm / delta_mean if abs(delta_mean) > 0 else float('nan')
                print(f"Inferred current->torque scale: {inferred_scale_med:.6e} Nm per current-unit (median-based)")
                print(f"Inferred (mean-based): {inferred_scale_mean:.6e} Nm per current-unit")

        else:
            print("No mass/lever provided; skipping torque computation. Provide --mass and --lever to compute Nm/current.")

        # Also report Present_Load if available
        obs = robot.get_observation()
        load_key = f"{args.motor}.load"
        if load_key in obs:
            print(f"Present_Load reported by motor: {obs[load_key]}")
        else:
            print("No Present_Load field available from motor in this setup.")

        print("Test complete. Use the inferred scale to compare with observer.current_to_torque_scale.")

    except Exception as e:
        print(f"Error during measurement: {e}")

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
