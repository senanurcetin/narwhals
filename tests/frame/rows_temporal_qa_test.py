"""Optional temporal scalar regression tests for Narwhals #4094."""

from __future__ import annotations

import pandas as pd
import pytest

import narwhals as nw


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("timezone", [None, "UTC", "Europe/Istanbul"])
def test_pandas_temporal_rows_keep_temporal_scalars(
    method: str,
    named: bool,  # noqa: FBT001
    timezone: str | None,
) -> None:
    timestamp = pd.Timestamp("2026-10-20 13:45:00", tz=timezone)
    duration = pd.Timedelta(days=2, hours=4, minutes=30)
    native = pd.DataFrame(
        {
            "when": pd.Series([timestamp, timestamp + pd.Timedelta(days=1)]),
            "duration": pd.Series([duration, duration + pd.Timedelta(hours=1)]),
        }
    )
    frame = nw.from_native(native, eager_only=True)
    values = list(getattr(frame, method)(named=named))
    first = tuple(values[0].values()) if named else values[0]
    second = tuple(values[1].values()) if named else values[1]

    assert first == (timestamp, duration)
    assert second == (
        timestamp + pd.Timedelta(days=1),
        duration + pd.Timedelta(hours=1),
    )
    assert type(first[0]) is pd.Timestamp
    assert type(first[1]) is pd.Timedelta
    assert type(second[0]) is pd.Timestamp
    assert type(second[1]) is pd.Timedelta
