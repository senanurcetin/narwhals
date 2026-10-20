"""Optional temporal scalar regression tests for Narwhals #4094."""

from __future__ import annotations

import numpy as np
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
    assert second == (timestamp + pd.Timedelta(days=1), duration + pd.Timedelta(hours=1))
    assert type(first[0]) is pd.Timestamp
    assert type(first[1]) is pd.Timedelta
    assert type(second[0]) is pd.Timestamp
    assert type(second[1]) is pd.Timedelta


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize(
    "kind", ["datetime", "timedelta", "datetime_utc", "datetime_istanbul"]
)
def test_missing_temporal_rows_are_none(
    method: str,
    named: bool,  # noqa: FBT001
    kind: str,
) -> None:
    """Pandas datetime/timedelta NaT must normalize to None in row outputs."""
    if kind == "datetime":
        values = pd.to_datetime(pd.Series(["2026-10-20", None], dtype="object"))
    elif kind == "timedelta":
        values = pd.to_timedelta(pd.Series(["2 days", None], dtype="object"))
    elif kind == "datetime_utc":
        values = pd.to_datetime(pd.Series(["2026-10-20", None], dtype="object"), utc=True)
    else:
        values = pd.to_datetime(
            pd.Series(["2026-10-20", None], dtype="object"), utc=True
        ).dt.tz_convert("Europe/Istanbul")

    native = pd.DataFrame(
        {"temporal": values, "number": pd.array([1, None], dtype="Int64")}
    )
    original = native.copy(deep=True)
    frame = nw.from_native(native, eager_only=True)
    result = list(getattr(frame, method)(named=named))
    first = tuple(result[0].values()) if named else result[0]
    second = tuple(result[1].values()) if named else result[1]

    assert first[1] == 1
    assert type(first[1]) is int
    assert second == (None, None)
    if kind == "timedelta":
        assert type(first[0]) is pd.Timedelta
    else:
        assert type(first[0]) is pd.Timestamp
    pd.testing.assert_frame_equal(native, original)


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("buffer_size", [1, 3, 512])
def test_missing_temporal_rows_across_batches(
    named: bool,  # noqa: FBT001
    buffer_size: int,
) -> None:
    """Ensure NaT becomes None at batch boundaries and across all batches."""
    values = pd.to_datetime(
        pd.Series(["2026-10-20" if i % 2 == 0 else None for i in range(1031)])
    )
    frame = nw.from_native(pd.DataFrame({"when": values}), eager_only=True)
    rows = list(frame.iter_rows(named=named, buffer_size=buffer_size))
    assert len(rows) == 1031

    for i in (1, 511, 513, 1029):
        record = rows[i]
        row = tuple(record.values()) if isinstance(record, dict) else record
        assert row == (None,)
    record = rows[512]
    row = tuple(record.values()) if isinstance(record, dict) else record
    assert type(row[0]) is pd.Timestamp


@pytest.mark.parametrize("kind", ["datetime", "timedelta"])
@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
def test_temporal_units_normalize_nat(
    kind: str,
    unit: str,
    method: str,
    named: bool,  # noqa: FBT001
) -> None:
    """Preserve real values and normalize NaT across datetime units."""
    if kind == "datetime":
        values = pd.Series(np.array(["2026-10-20", "NaT"], dtype=f"datetime64[{unit}]"))
        expected_type = pd.Timestamp
    else:
        values = pd.Series(np.array([1, "NaT"], dtype=f"timedelta64[{unit}]"))
        expected_type = pd.Timedelta

    frame = nw.from_native(pd.DataFrame({"temporal": values}), eager_only=True)
    result = list(getattr(frame, method)(named=named))
    first = tuple(result[0].values()) if named else result[0]
    second = tuple(result[1].values()) if named else result[1]

    assert type(first[0]) is expected_type
    assert second == (None,)
