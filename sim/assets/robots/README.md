# Robot Assets

Place custom robot directories here. Each robot needs:

```
robots/<robot_name>/
    robot.urdf          # URDF model file
    metadata.json       # Robot metadata (see below)
```

## metadata.json Schema

```json
{
    "name": "my_robot",
    "foot_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    "base_body_name": "base",
    "standing_height": 0.34,
    "num_legs": 4
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Display name for the robot |
| `foot_body_names` | string[] | URDF link names for each foot (used for contact rewards) |
| `base_body_name` | string | URDF link name for the robot's base/torso |
| `standing_height` | float | Nominal standing height in meters (used for height reward) |
| `num_legs` | int | Number of legs (currently only 4 supported) |

### Notes

- DOF count is auto-detected from the URDF (counts revolute/continuous/prismatic joints)
- Foot link names must exactly match the URDF link names
- The base body name is used for termination detection (illegal base contact)
- Standing height should be measured from ground to base link origin when standing
