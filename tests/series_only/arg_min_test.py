from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.mark.parametrize(("col", "expected"), [("a", 0), ("b", 0), ("z", 0)])
def test_arg_min_series(
    constructor_eager: ConstructorEager,
    col: str,
    expected: float,
    request: pytest.FixtureRequest,
) -> None:
    if "modin" in str(constructor_eager):
        # TODO(unassigned): bug in modin?
        return
    if "cudf" in str(constructor_eager):
        # not implemented yet
        request.applymarker(pytest.mark.xfail)
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    series = nw.maybe_set_index(series, index=[1, 0, 9])  # type: ignore[arg-type]
    result = series.arg_min()
    assert_equal_data({col: [result]}, {col: [expected]})


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([], None, id="empty"),
        pytest.param([None, None], None, id="all-null"),
        pytest.param([None], None, id="single-null"),
        pytest.param([None, 3.0, 1.0, None], 2, id="mixed-null"),
    ],
)
@pytest.mark.parametrize("as_integer", [False, True], ids=["float64", "int64"])
def test_arg_min_nullable_series(
    constructor_eager: ConstructorEager,
    values: list[float | None],
    expected: int | None,
    request: pytest.FixtureRequest,
    *,
    as_integer: bool,
) -> None:
    if "modin" in str(constructor_eager):
        pytest.skip("Modin arg_min is not supported by existing tests")
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    if as_integer and "pandas" in str(constructor_eager):
        # Pandas' plain int64 dtype cannot represent nulls.
        import pandas as pd

        dtype = "int64[pyarrow]" if "pyarrow" in str(constructor_eager) else "Int64"
        native = pd.DataFrame({"a": pd.Series(values, dtype=dtype)})
        series = nw.from_native(native, eager_only=True)["a"]
    else:
        series = nw.from_native(constructor_eager({"a": values}), eager_only=True)[
            "a"
        ].cast(nw.Int64() if as_integer else nw.Float64())
    assert series.arg_min() == expected
