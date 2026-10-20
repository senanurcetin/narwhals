from __future__ import annotations

import json

import pandas as pd
import pytest

import narwhals as nw


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
def test_nullable_rows_are_python_native(
    method: str, named: bool,  # noqa: FBT001
) -> None:
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


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("buffer_size", [0, 1, 3, 7, 512])
def test_nullable_iter_rows_batch_boundaries(
    named: bool, buffer_size: int,  # noqa: FBT001
) -> None:
    native = pd.DataFrame(
        {
            "count": pd.array([1, None] * 515, dtype="Int64"),
            "flag": pd.array([True, None] * 515, dtype="boolean"),
        }
    )
    frame = nw.from_native(native, eager_only=True)
    rows = list(frame.iter_rows(named=named, buffer_size=buffer_size))
    assert len(rows) == 1030
    for index in (0, 1, 2, 511, 512, 1028, 1029):
        expected = (1, True) if index % 2 == 0 else (None, None)
        actual = tuple(rows[index].values()) if named else rows[index]
        assert actual == expected


@pytest.mark.parametrize("buffer_size", [0, 1, 2, 5])
@pytest.mark.parametrize("named", [False, True])
def test_iter_rows_zero_columns_across_batches(
    buffer_size: int, named: bool,  # noqa: FBT001
) -> None:
    frame = nw.from_native(pd.DataFrame(index=range(13)), eager_only=True)
    expected = [{}] * 13 if named else [()] * 13
    assert list(frame.iter_rows(named=named, buffer_size=buffer_size)) == expected


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
def test_nullable_categorical_rows_match(method: str, named: bool) -> None:  # noqa: FBT001
    native = pd.DataFrame(
        {
            "category": pd.Categorical(["a", None, "b"]),
            "text": pd.array(["x", None, "y"], dtype="string"),
        }
    )
    original = native.copy(deep=True)
    df = nw.from_native(native, eager_only=True)
    result = list(getattr(df, method)(named=named))
    expected = (
        [
            {"category": "a", "text": "x"},
            {"category": None, "text": None},
            {"category": "b", "text": "y"},
        ]
        if named
        else [("a", "x"), (None, None), ("b", "y")]
    )
    assert result == expected
    json.dumps(result)
    pd.testing.assert_frame_equal(native, original)


def test_pandas_named_rows_zero_columns_unchanged() -> None:
    frame = nw.from_native(pd.DataFrame(index=range(3)), eager_only=True)
    assert frame.rows(named=True) == []
