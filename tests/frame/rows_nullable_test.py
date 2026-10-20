from __future__ import annotations

import json

import pandas as pd
import pytest

import narwhals as nw


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
def test_nullable_rows_are_python_native(method: str, named: bool) -> None:  # noqa: FBT001
    native = pd.DataFrame(
        {
            "count": pd.array([1, None], dtype="Int64"),
            "flag": pd.array([True, None], dtype="boolean"),
            "amount": pd.array([1.5, None], dtype="Float64"),
        }
    )
    df = nw.from_native(native, eager_only=True)
    rows = list(getattr(df, method)(named=named))

    expected = (
        [
            {"count": 1, "flag": True, "amount": 1.5},
            {"count": None, "flag": None, "amount": None},
        ]
        if named
        else [(1, True, 1.5), (None, None, None)]
    )
    assert rows == expected
    first = tuple(rows[0].values()) if named else rows[0]
    missing = tuple(rows[1].values()) if named else rows[1]
    assert tuple(type(value) for value in first) == (int, bool, float)
    assert all(value is None for value in missing)
    json.dumps(rows)


@pytest.mark.parametrize("named", [False, True])
def test_plain_rows_unchanged(named: bool) -> None:  # noqa: FBT001
    native = pd.DataFrame({"text": ["a", "b"], "value": [3, 4]})
    df = nw.from_native(native, eager_only=True)
    expected = (
        [{"text": "a", "value": 3}, {"text": "b", "value": 4}]
        if named
        else [("a", 3), ("b", 4)]
    )
    assert df.rows(named=named) == expected
    assert list(df.iter_rows(named=named)) == expected


@pytest.mark.parametrize(
    ("method", "named"),
    [("rows", False), ("iter_rows", False), ("iter_rows", True)],
)
def test_zero_columns_with_rows(method: str, named: bool) -> None:  # noqa: FBT001
    df = nw.from_native(pd.DataFrame(index=[0, 1, 2]), eager_only=True)
    expected = [{}, {}, {}] if named else [(), (), ()]
    assert list(getattr(df, method)(named=named)) == expected


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
def test_no_rows(method: str, named: bool) -> None:  # noqa: FBT001
    native = pd.DataFrame({"value": pd.Series(dtype="Int64")})
    df = nw.from_native(native, eager_only=True)
    assert list(getattr(df, method)(named=named)) == []
