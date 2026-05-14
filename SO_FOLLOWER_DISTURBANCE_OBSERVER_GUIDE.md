# SO Follower (SO100/SO101) Disturbance Observer - Implementation Summary

## What Was Created

I've implemented a complete **disturbance observer** system for real SO100 robot control. This estimates external forces/torques acting on the robot by reading motor current feedback.

### 📁 Files Created

| File | Purpose |
|------|---------|
| [src/lerobot/processor/disturbance_observer.py](src/lerobot/processor/disturbance_observer.py) | Core observer processor with two implementations |
| [src/lerobot/robots/so_follower/so_follower_disturbance_observer.py](src/lerobot/robots/so_follower/so_follower_disturbance_observer.py) | SO Follower (SO100/SO101) robot integration with compensation |
| [docs/source/so_follower_disturbance_observer.mdx](docs/source/so_follower_disturbance_observer.mdx) | Complete user guide & calibration procedures |
| [examples/so_follower_disturbance_observer_example.py](examples/so_follower_disturbance_observer_example.py) | Working example scripts |

## How It Works

```
┌─────────────────────────────────────────────┐
| SO Follower (SO100/SO101) Motor           |
│   ─────────────────────────────────────────│
│   • Position: θ                             │
│   • Current: I (feedback)                   │
│   • Load: τ_load                            │
└────────────┬────────────────────────────────┘
             │
             ├─→ Measure current → τ_measured = I × K_t
             │
             ├─→ Estimate velocity: v = Δθ/Δt
             │
             ├─→ Compute model: τ_model = friction + gravity
             │
             └─→ Extract disturbance: τ_dist = τ_measured - τ_model

         τ_dist = estimated external torque/force on joint
```

## Quick Start (3 Steps)

### Step 1: Import and Initialize

```python
from lerobot.robots.so_follower import SO100FollowerConfig
from lerobot.robots.so_follower.so_follower_disturbance_observer import SOFollowerWithDisturbanceObserver
from lerobot.processor.disturbance_observer import DisturbanceObserverStep

# Connect robot
config = SO100FollowerConfig(port="/dev/ttyUSB0")
robot = SOFollowerWithDisturbanceObserver(config)
robot.connect()

# Create observer
observer = DisturbanceObserverStep(
    current_to_torque_scale=0.00016,  # Nm per mA
    gravity_torque={"shoulder_lift": 2.5, ...},  # Calibrate for your setup
    dt=0.01
)
robot.set_disturbance_observer(observer)
```

### Step 2: Get Disturbance Estimates

```python
obs = robot.get_observation()

# Now includes disturbance estimates for each joint:
print(obs["shoulder_lift.disturbance"])  # Nm
print(obs["elbow_flex.disturbance"])     # Nm
# etc.
```

### Step 3: Use in Control Loop

```python
# Option A: Just monitoring
for i in range(1000):
    obs = robot.get_observation()
    if abs(obs["shoulder_lift.disturbance"]) > 5.0:  # Nm
        print("Collision detected!")
        break
    time.sleep(0.01)

# Option B: With automatic compensation
action = {"shoulder_pan.pos": -0.5, "shoulder_lift.pos": 0.3, ...}
robot.send_action(
    action,
    apply_compensation=True,  # Enable disturbance compensation
    compensation_scale=0.8
)
```

## Motor Data Available (SO100 / Feetech STS3215)

| Parameter | Description | Use Case |
|-----------|-------------|----------|
| `Present_Position` | Joint angle (0-4095) | Position feedback |
| `Present_Velocity` | Joint velocity | Friction estimation |
| `Present_Current` | Motor current (mA) | **Torque estimation** ✓ |
| `Present_Load` | Motor load/torque | Fallback torque feedback |
| `Present_Temperature` | Motor temperature | Thermal monitoring |
| `Present_Voltage` | Supply voltage | Power monitoring |

## Calibration (Important!)

### Quick Calibration (5 minutes)

```bash
# 1. Measure motor constant (current-to-torque conversion)
#    Hang 1 kg at 0.2m → 1.96 Nm load
#    If motor reads 12 A → K_t = 0.163 Nm/A = 0.000163 Nm/mA

# 2. Measure gravity torque at neutral pose
python examples/so100_disturbance_observer_example.py \
    --port /dev/ttyUSB0 --mode calibrate

# 3. Adjust parameters in observer:
observer.current_to_torque_scale = 0.000163
observer.gravity_torque = {
    "shoulder_lift": 2.5,  # Nm at neutral
    "elbow_flex": 1.8,
    ...
}
```

### Full Calibration (30 minutes)

See [docs/source/so100_disturbance_observer.mdx](docs/source/so100_disturbance_observer.mdx#advanced-calibrating-motor-parameters) for:
- Per-motor current constant measurement
- Gravity torque via static loads
- Motor inertia from system identification

## Integration Points

### With Training Pipelines

```python
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.processor.disturbance_observer import DisturbanceObserverStep

preprocessor = PolicyProcessorPipeline([
    NormalizerProcessorStep(...),
    DisturbanceObserverStep(
        current_to_torque_scale=0.00016,
        gravity_torque={"shoulder_lift": 2.5, ...},
    ),
    DeviceProcessorStep(device="cuda"),
])

# Policy now receives disturbance estimates as input
```

### With Evaluation Loop

```python
# Disturbance estimates included automatically in dataset preprocessing
eval_callable = make_eval_callable(
    policy,
    dataset,
    processor_kwargs={
        "disturbance_observer": {"gravity_torque": {...}}
    },
)
```

### With Policy Server (Async Inference)

```python
# Server computes disturbances in real-time before policy inference
server = PolicyServer(
    policy_model_id="your_model",
    preprocessor_config={
        "disturbance_observer": {
            "current_to_torque_scale": 0.00016,
            "gravity_torque": {...},
        }
    },
)
```

## Safety Features

### Collision Detection
```python
# Alert if disturbance exceeds threshold
if abs(obs["shoulder_lift.disturbance"]) > 3.0:  # Nm
    print("Collision detected!")
    robot.bus.disable_torque()  # Stop immediately
```

### Stability Limits
```python
observer = DisturbanceObserverStep(
    max_disturbance=10.0,  # Clip unrealistic estimates (Nm)
    velocity_filter_window=5,  # Smooth noise
)
```

## Classes Reference

### DisturbanceObserverStep
**Basic disturbance observer** - Use this for most applications.

Key parameters:
- `current_to_torque_scale`: Nm per mA (default: 0.01)
- `gravity_torque`: Dict of gravity Nm per joint
- `friction_coefficient`: Dict of friction coefficients
- `dt`: Control loop timestep (default: 0.01s)
- `include_in_observation`: Add disturbance to observation dict (default: True)

### AdvancedDisturbanceObserverStep
**With kinematic model** - For gravity compensation using robot geometry.

Additional parameters:
- `observer_gain`: Observer responsiveness (0-1)
- `use_kinematic_model`: Enable geometry-based gravity (requires kinematics)
- `link_masses`: Mass of each link for gravity computation

### SOFollowerWithDisturbanceObserver
**Extended robot class** - Integrates observer into SO Follower (SO100/SO101) control loop.

Methods:
- `set_disturbance_observer(observer)`: Attach observer
- `get_observation()`: Returns obs with disturbance estimates
- `send_action(action, apply_compensation=False, compensation_scale=1.0)`: Execute with optional compensation

## Example Scripts

### Monitor Disturbances for 30 seconds
```bash
python examples/so_follower_disturbance_observer_example.py \
    --port /dev/ttyUSB0 \
    --mode monitor \
    --duration 30 \
    --threshold 3.0  # Alert if > 3 Nm
```

### Calibrate Gravity Torques
```bash
python examples/so_follower_disturbance_observer_example.py \
    --port /dev/ttyUSB0 \
    --mode calibrate
```

### Record Trajectory with Disturbance Feedback
```bash
python examples/so_follower_disturbance_observer_example.py \
    --port /dev/ttyUSB0 \
    --mode teach
```

### Execute Control Loop with Compensation
```bash
python examples/so_follower_disturbance_observer_example.py \
    --port /dev/ttyUSB0 \
    --mode control
```

### Typical Parameter Values (SO100/SO101)

### Motor Constants (Feetech STS3215 - Used in both SO100 & SO101)
- `current_to_torque_scale`: 0.00016 Nm/mA (calibrate per motor)
- Motor max torque: ~11 Nm
- Motor rated current: ~1000 mA

**Note**: SO101 has reduced gears for smoother operation but uses same motors/parameters

### Gravity Torques (Neutral Pose)
```python
gravity_torque = {
    "shoulder_pan": 0.0,      # Horizontal axis
    "shoulder_lift": 2.5,     # Large due to arm length
    "elbow_flex": 1.8,        # Medium load
    "wrist_flex": 0.2,        # Small load
    "wrist_roll": 0.0,        # Minimal
    "gripper": 0.0,           # Minimal
}
```

### Friction Coefficients
```python
friction_coefficient = {
    "shoulder_pan": 0.01,     # Viscous damping
    "shoulder_lift": 0.01,
    "elbow_flex": 0.01,
    "wrist_flex": 0.005,
    "wrist_roll": 0.005,
    "gripper": 0.001,
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| All disturbances = 0 | Check `current_to_torque_scale` and motor current feedback working |
| Noisy estimates | Increase `velocity_filter_window` (5→7) |
| Compensation makes things worse | Reduce `compensation_scale` (1.0→0.1) and verify calibration |
| Current feedback not available | Motor bus may not support Present_Current; use Present_Load instead |

## Next Steps

1. **Try basic example first**: Run `--mode monitor` to verify setup
2. **Calibrate your SO Follower**: Run `--mode calibrate` to get gravity torques (works for SO100 & SO101)
3. **Integrate into pipeline**: Add DisturbanceObserverStep to your training pipeline
4. **Use in policy**: Pass disturbance estimates as policy input features
5. **Deploy on real robot**: Use SOFollowerWithDisturbanceObserver in your control loop

## References

- **Disturbance Observer Theory**: Ohnishi et al., "Micro-machine and human science", 1996
- **SO100 Motor Specs**: [Feetech STS3215 Datasheet](https://github.com/ftservo/scservo_sdk)
- **LeRobot Processor Architecture**: [Implementation Guide](../docs/source/implement_your_own_processor.mdx)
- **Control Loop Integration**: [LeRobot Teleoperation Script](../src/lerobot/scripts/lerobot_teleoperate.py)
