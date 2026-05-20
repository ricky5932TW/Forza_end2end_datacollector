"""Forza Horizon Data Out packet parser.

Example:
    telemetry = parse_packet(payload, packet_format="horizon")
    speed_mps = telemetry["Speed"]
    steering = telemetry["Steer"]
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldSpec:
    """One field in the packed Forza telemetry layout."""

    name: str
    data_type: str


TYPE_FORMATS = {
    "s32": "i",
    "u32": "I",
    "f32": "f",
    "u16": "H",
    "u8": "B",
    "s8": "b",
    "bytes12": "12s",
}


HORIZON_FIELDS = (
    FieldSpec("IsRaceOn", "s32"),
    FieldSpec("TimestampMS", "u32"),
    FieldSpec("EngineMaxRpm", "f32"),
    FieldSpec("EngineIdleRpm", "f32"),
    FieldSpec("CurrentEngineRpm", "f32"),
    FieldSpec("AccelerationX", "f32"),
    FieldSpec("AccelerationY", "f32"),
    FieldSpec("AccelerationZ", "f32"),
    FieldSpec("VelocityX", "f32"),
    FieldSpec("VelocityY", "f32"),
    FieldSpec("VelocityZ", "f32"),
    FieldSpec("AngularVelocityX", "f32"),
    FieldSpec("AngularVelocityY", "f32"),
    FieldSpec("AngularVelocityZ", "f32"),
    FieldSpec("Yaw", "f32"),
    FieldSpec("Pitch", "f32"),
    FieldSpec("Roll", "f32"),
    FieldSpec("NormalizedSuspensionTravelFrontLeft", "f32"),
    FieldSpec("NormalizedSuspensionTravelFrontRight", "f32"),
    FieldSpec("NormalizedSuspensionTravelRearLeft", "f32"),
    FieldSpec("NormalizedSuspensionTravelRearRight", "f32"),
    FieldSpec("TireSlipRatioFrontLeft", "f32"),
    FieldSpec("TireSlipRatioFrontRight", "f32"),
    FieldSpec("TireSlipRatioRearLeft", "f32"),
    FieldSpec("TireSlipRatioRearRight", "f32"),
    FieldSpec("WheelRotationSpeedFrontLeft", "f32"),
    FieldSpec("WheelRotationSpeedFrontRight", "f32"),
    FieldSpec("WheelRotationSpeedRearLeft", "f32"),
    FieldSpec("WheelRotationSpeedRearRight", "f32"),
    FieldSpec("WheelOnRumbleStripFrontLeft", "s32"),
    FieldSpec("WheelOnRumbleStripFrontRight", "s32"),
    FieldSpec("WheelOnRumbleStripRearLeft", "s32"),
    FieldSpec("WheelOnRumbleStripRearRight", "s32"),
    FieldSpec("WheelInPuddleDepthFrontLeft", "f32"),
    FieldSpec("WheelInPuddleDepthFrontRight", "f32"),
    FieldSpec("WheelInPuddleDepthRearLeft", "f32"),
    FieldSpec("WheelInPuddleDepthRearRight", "f32"),
    FieldSpec("SurfaceRumbleFrontLeft", "f32"),
    FieldSpec("SurfaceRumbleFrontRight", "f32"),
    FieldSpec("SurfaceRumbleRearLeft", "f32"),
    FieldSpec("SurfaceRumbleRearRight", "f32"),
    FieldSpec("TireSlipAngleFrontLeft", "f32"),
    FieldSpec("TireSlipAngleFrontRight", "f32"),
    FieldSpec("TireSlipAngleRearLeft", "f32"),
    FieldSpec("TireSlipAngleRearRight", "f32"),
    FieldSpec("TireCombinedSlipFrontLeft", "f32"),
    FieldSpec("TireCombinedSlipFrontRight", "f32"),
    FieldSpec("TireCombinedSlipRearLeft", "f32"),
    FieldSpec("TireCombinedSlipRearRight", "f32"),
    FieldSpec("SuspensionTravelMetersFrontLeft", "f32"),
    FieldSpec("SuspensionTravelMetersFrontRight", "f32"),
    FieldSpec("SuspensionTravelMetersRearLeft", "f32"),
    FieldSpec("SuspensionTravelMetersRearRight", "f32"),
    FieldSpec("CarOrdinal", "s32"),
    FieldSpec("CarClass", "s32"),
    FieldSpec("CarPerformanceIndex", "s32"),
    FieldSpec("DrivetrainType", "s32"),
    FieldSpec("NumCylinders", "s32"),
    FieldSpec("HorizonUnknownBytes", "bytes12"),
    FieldSpec("PositionX", "f32"),
    FieldSpec("PositionY", "f32"),
    FieldSpec("PositionZ", "f32"),
    FieldSpec("Speed", "f32"),
    FieldSpec("Power", "f32"),
    FieldSpec("Torque", "f32"),
    FieldSpec("TireTempFrontLeft", "f32"),
    FieldSpec("TireTempFrontRight", "f32"),
    FieldSpec("TireTempRearLeft", "f32"),
    FieldSpec("TireTempRearRight", "f32"),
    FieldSpec("Boost", "f32"),
    FieldSpec("Fuel", "f32"),
    FieldSpec("DistanceTraveled", "f32"),
    FieldSpec("BestLap", "f32"),
    FieldSpec("LastLap", "f32"),
    FieldSpec("CurrentLap", "f32"),
    FieldSpec("CurrentRaceTime", "f32"),
    FieldSpec("LapNumber", "u16"),
    FieldSpec("RacePosition", "u8"),
    FieldSpec("Accel", "u8"),
    FieldSpec("Brake", "u8"),
    FieldSpec("Clutch", "u8"),
    FieldSpec("HandBrake", "u8"),
    FieldSpec("Gear", "u8"),
    FieldSpec("Steer", "s8"),
    FieldSpec("NormalizedDrivingLine", "s8"),
    FieldSpec("NormalizedAIBrakeDifference", "s8"),
)

HORIZON_FIELD_NAMES = tuple(field.name for field in HORIZON_FIELDS)
HORIZON_STRUCT_FORMAT = "<" + "".join(TYPE_FORMATS[field.data_type] for field in HORIZON_FIELDS)
HORIZON_PACKET_SIZE = struct.calcsize(HORIZON_STRUCT_FORMAT)


class PacketSizeError(ValueError):
    """Raised when a UDP packet is too short for the selected parser."""


def parse_horizon_packet(packet: bytes) -> dict[str, Any]:
    """Parse one Horizon UDP payload into a plain dictionary."""

    if len(packet) < HORIZON_PACKET_SIZE:
        raise PacketSizeError(f"Horizon packet needs {HORIZON_PACKET_SIZE} bytes, got {len(packet)}")

    values = struct.unpack_from(HORIZON_STRUCT_FORMAT, packet)
    telemetry: dict[str, Any] = {}
    for field, value in zip(HORIZON_FIELDS, values):
        telemetry[field.name] = value.hex() if isinstance(value, bytes) else value
    return telemetry


def parse_packet(packet: bytes, packet_format: str = "horizon") -> dict[str, Any]:
    """Parse one UDP payload with the selected packet format.

    Example:
        telemetry = parse_packet(payload)
        throttle = telemetry["Accel"]
    """

    if packet_format.lower() != "horizon":
        raise ValueError(f"unsupported packet format: {packet_format}")
    return parse_horizon_packet(packet)


def field_names(packet_format: str = "horizon") -> tuple[str, ...]:
    """Return all parser field names for CSV/JSONL output."""

    if packet_format.lower() != "horizon":
        raise ValueError(f"unsupported packet format: {packet_format}")
    return HORIZON_FIELD_NAMES


def expected_packet_size(packet_format: str = "horizon") -> int:
    """Return the minimum number of bytes required for a packet layout."""

    if packet_format.lower() != "horizon":
        raise ValueError(f"unsupported packet format: {packet_format}")
    return HORIZON_PACKET_SIZE
