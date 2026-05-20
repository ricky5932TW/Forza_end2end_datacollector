import struct
import unittest

from packet_format import HORIZON_FIELDS, HORIZON_PACKET_SIZE, HORIZON_STRUCT_FORMAT, PacketSizeError
from packet_format import parse_horizon_packet


def fake_horizon_packet() -> bytes:
    values = []
    for field in HORIZON_FIELDS:
        if field.name == "IsRaceOn":
            values.append(1)
        elif field.name == "TimestampMS":
            values.append(123456)
        elif field.name == "CurrentEngineRpm":
            values.append(3456.5)
        elif field.name == "Speed":
            values.append(42.25)
        elif field.name == "Gear":
            values.append(4)
        elif field.name == "Accel":
            values.append(200)
        elif field.name == "Brake":
            values.append(12)
        elif field.name == "Steer":
            values.append(-8)
        elif field.data_type == "bytes12":
            values.append(bytes(range(12)))
        elif field.data_type in {"s32", "u32", "u16", "u8", "s8"}:
            values.append(0)
        else:
            values.append(0.0)
    return struct.pack(HORIZON_STRUCT_FORMAT, *values)


class HorizonPacketTests(unittest.TestCase):
    def test_parse_fake_packet(self) -> None:
        packet = fake_horizon_packet()
        self.assertEqual(len(packet), HORIZON_PACKET_SIZE)

        telemetry = parse_horizon_packet(packet)

        self.assertEqual(telemetry["IsRaceOn"], 1)
        self.assertEqual(telemetry["TimestampMS"], 123456)
        self.assertAlmostEqual(telemetry["CurrentEngineRpm"], 3456.5)
        self.assertAlmostEqual(telemetry["Speed"], 42.25)
        self.assertEqual(telemetry["Gear"], 4)
        self.assertEqual(telemetry["Accel"], 200)
        self.assertEqual(telemetry["Brake"], 12)
        self.assertEqual(telemetry["Steer"], -8)
        self.assertEqual(telemetry["HorizonUnknownBytes"], "000102030405060708090a0b")

    def test_accepts_trailing_bytes(self) -> None:
        telemetry = parse_horizon_packet(fake_horizon_packet() + b"extra")
        self.assertAlmostEqual(telemetry["Speed"], 42.25)

    def test_short_packet_raises(self) -> None:
        with self.assertRaises(PacketSizeError):
            parse_horizon_packet(b"short")


if __name__ == "__main__":
    unittest.main()
