# ROS Interfaces

The final integrated application requires ROS for mission-time synchronization and publishes the vision outputs used elsewhere in the UAV software stack.

## Required time input

### `/server_time`

- Direction: **subscribe**
- Type in the integrated code: `std_msgs/String`
- Role: authoritative mission/server clock

The combined application intentionally does not fall back to the local PC clock for required mission timestamps. At startup it waits for a valid server-time source and stops startup if the source is unavailable within the configured timeout.

## QR output

### `/kamikaze_bilgisi`

- Direction: **publish**
- Type: `std_msgs/String`
- Payload: JSON-encoded kamikaze start time, end time, and decoded QR text

## LOCK / tracking outputs

### `/target_data`

- Direction: **publish**
- Role: selected/focused target tracking data used by downstream target-control logic

### `/hedef_piksel`

- Direction: **publish**
- Role: target pixel information

### `/kilitlenme_bilgisi`

- Direction: **publish**
- Role: successful lock event information

## Configuration

`target_data_topic` and `lock_topic` are exposed through `config.yaml`. Other interfaces use the final competition defaults in `main.py`.

## QR MPC note

The integrated source still contains the earlier `/qr/target_error` support code, but its publisher is intentionally disabled in the final combined configuration. It is documented as legacy functionality rather than an active public interface.
