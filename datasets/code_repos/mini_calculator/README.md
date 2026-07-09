# mini_calculator

A tiny, real Python package used as Module 17's sample repo for the local coding assistant
labs. `calculator.py` has a genuine bug: `average([])` raises `ZeroDivisionError` instead of
returning `0.0`, and `tests/test_calculator.py::test_average_of_empty_list_should_return_zero`
fails against the unpatched code - the labs propose, validate, and apply a real patch to fix it,
then re-run the real test suite to confirm the fix.
