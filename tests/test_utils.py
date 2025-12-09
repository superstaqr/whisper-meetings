"""Tests for utility functions."""

import pytest

from whisper_meetings.utils import calculate_overlap, format_timestamp


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_simple_timestamp(self):
        result = format_timestamp(0, 10)
        assert result == "[00:00:00.000 -> 00:00:10.000]"

    def test_timestamp_with_milliseconds(self):
        result = format_timestamp(1.5, 2.75)
        assert result == "[00:00:01.500 -> 00:00:02.750]"

    def test_timestamp_with_minutes(self):
        result = format_timestamp(65.0, 130.0)
        assert result == "[00:01:05.000 -> 00:02:10.000]"

    def test_timestamp_with_hours(self):
        result = format_timestamp(3661.123, 7322.456)
        assert result == "[01:01:01.123 -> 02:02:02.456]"


class TestCalculateOverlap:
    """Tests for calculate_overlap function."""

    def test_no_overlap(self):
        assert calculate_overlap(0, 10, 20, 30) == 0.0

    def test_full_overlap(self):
        assert calculate_overlap(5, 10, 5, 10) == 5.0

    def test_partial_overlap_start(self):
        assert calculate_overlap(0, 10, 5, 15) == 5.0

    def test_partial_overlap_end(self):
        assert calculate_overlap(5, 15, 0, 10) == 5.0

    def test_contained_interval(self):
        assert calculate_overlap(5, 8, 0, 20) == 3.0

    def test_containing_interval(self):
        assert calculate_overlap(0, 20, 5, 8) == 3.0

    def test_adjacent_intervals(self):
        assert calculate_overlap(0, 10, 10, 20) == 0.0
