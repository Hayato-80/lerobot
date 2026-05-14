#!/usr/bin/env python3
"""Quick test to verify action/observation space configuration."""
import json
from pathlib import Path

config_path = Path("src/lerobot/configs/learner_config_so101.json")
with open(config_path) as f:
    config = json.load(f)

print("=" * 60)
print("ACTION SPACE VERIFICATION")
print("=" * 60)

# Policy output features
policy_action_shape = config["policy"]["output_features"]["action"]["shape"]
print(f"\n✓ Policy output_features.action.shape: {policy_action_shape}")
print(f"  Expected: [6] (6 motors)")
assert policy_action_shape == [6], f"ERROR: Policy action shape should be [6], got {policy_action_shape}"

# Policy num_discrete_actions
num_discrete = config["policy"]["num_discrete_actions"]
print(f"\n✓ Policy num_discrete_actions: {num_discrete}")
print(f"  Expected: null (continuous SAC)")
assert num_discrete is None, f"ERROR: Should be null for continuous, got {num_discrete}"

# Env features
env_action_shape = config["env"]["features"]["action"]["shape"]
print(f"\n✓ Env features.action.shape: {env_action_shape}")
print(f"  Expected: [6]")
assert env_action_shape == [6], f"ERROR: Env action shape should be [6], got {env_action_shape}"

# Dataset stats
action_min = config["policy"]["dataset_stats"]["action"]["min"]
action_max = config["policy"]["dataset_stats"]["action"]["max"]
print(f"\n✓ Action normalization stats:")
print(f"  min: {action_min} (len={len(action_min)})")
print(f"  max: {action_max} (len={len(action_max)})")
assert len(action_min) == 6, f"ERROR: Action min should have 6 values"
assert len(action_max) == 6, f"ERROR: Action max should have 6 values"

# Control mode
control_mode = config["env"]["processor"]["control_mode"]
print(f"\n✓ Control mode: {control_mode}")
print(f"  Expected: leader (direct joint positions)")
assert control_mode == "leader", f"ERROR: Control mode should be 'leader', got {control_mode}"

# Reset pose
reset_pose = config["env"]["processor"]["reset"]["fixed_reset_joint_positions"]
print(f"\n✓ Reset pose: {reset_pose} (len={len(reset_pose)})")
print(f"  Expected: 6 joint positions")
assert len(reset_pose) == 6, f"ERROR: Reset pose should have 6 values"

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED!")
print("=" * 60)
print("\nConfiguration is now correct for 6-motor SO101 robot in leader mode.")
print("Policy will output 6D continuous actions (no discrete gripper).")
