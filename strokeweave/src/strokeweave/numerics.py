"""Small, exact, full-rank least-squares utilities; no optional dependencies.

Bareiss elimination and Neumann/supersolution bounds are established numerical
methods, not StrokeWeave inventions. This module certifies the exact rational
problem supplied to it, not the fidelity of measurements or arbitrary software.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any, Iterable


F = Fraction
TOLERANCE = F(1, 100_000_000)
NEUMANN_TERMS = 3
MAX_COLUMNS = 64
MAX_NUMBER_BITS = 16_384
MAX_DECIMAL_EXPONENT = 4_096


class NumericInputError(ValueError):
    """A declared input contract failed; no answer has been certified."""


def _fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise NumericInputError("boolean_number")
    if isinstance(value, F):
        parsed = value
    elif isinstance(value, int):
        parsed = F(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise NumericInputError("nonfinite_number")
        # A float denotes its exact binary rational value, not a guessed decimal.
        parsed = F(value)
    elif isinstance(value, str):
        if len(value) > 4096:
            raise NumericInputError("numeric_token_too_long")
        exponent = re.search(r"[eE]([+-]?\d+)$", value.strip())
        if exponent and (len(exponent.group(1).lstrip("+-")) > 5 or
                         abs(int(exponent.group(1))) > MAX_DECIMAL_EXPONENT):
            raise NumericInputError("numeric_exponent_limit_exceeded")
        try:
            parsed = F(value)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise NumericInputError("invalid_or_nonfinite_number") from exc
    else:
        raise NumericInputError("unsupported_number_type")
    if max(parsed.numerator.bit_length(), parsed.denominator.bit_length()) > MAX_NUMBER_BITS:
        raise NumericInputError("numeric_bit_limit_exceeded")
    return parsed


def _sequence(value: Any, reason: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise NumericInputError(reason)
    return tuple(value)


def _vector(value: Any, reason: str = "invalid_vector") -> tuple[Fraction, ...]:
    return tuple(_fraction(x) for x in _sequence(value, reason))


def _matrix(value: Any) -> tuple[tuple[Fraction, ...], ...]:
    rows = tuple(_vector(row, "invalid_matrix_row") for row in _sequence(value, "invalid_matrix"))
    if not rows or not rows[0]:
        raise NumericInputError("empty_matrix")
    if any(len(row) != len(rows[0]) for row in rows):
        raise NumericInputError("ragged_matrix")
    return rows


def _digest(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _source_hash(X: tuple, y: tuple, source_id: str) -> str:
    return _digest({"schema": "strokeweave-exact-ls-v1", "source_id": source_id,
                    "design": [[str(x) for x in row] for row in X],
                    "response": [str(x) for x in y]})


def _statistics_hash(H: tuple, h: tuple, c: F, scales: tuple, source_hash: str) -> str:
    return _digest({"source_hash": source_hash,
                    "gram": [[str(x) for x in row] for row in H],
                    "rhs": [str(x) for x in h], "norm_y2": str(c),
                    "scales": [str(x) for x in scales]})


def _dot(a: Iterable[F], b: Iterable[F]) -> F:
    return sum((x * y for x, y in zip(a, b)), F(0))


def _matvec(A: tuple, x: tuple) -> tuple[F, ...]:
    return tuple(_dot(row, x) for row in A)


def _require_positive_definite(H: tuple) -> None:
    """Exact LDL pivots validate positive definiteness, without tolerances."""
    p = len(H)
    L = [[F(0) for _ in range(p)] for _ in range(p)]
    D = []
    for i in range(p):
        if len(H[i]) != p or any(H[i][j] != H[j][i] for j in range(p)):
            raise NumericInputError("nonsymmetric_gram")
        pivot = H[i][i] - sum((L[i][k] ** 2 * D[k] for k in range(i)), F(0))
        if pivot <= 0:
            raise NumericInputError("design_not_full_column_rank")
        D.append(pivot)
        L[i][i] = F(1)
        for j in range(i + 1, p):
            L[j][i] = (H[j][i] - sum((L[j][k] * L[i][k] * D[k]
                                     for k in range(i)), F(0))) / pivot


@dataclass(frozen=True)
class PreparedProblem:
    design: tuple[tuple[F, ...], ...]
    response: tuple[F, ...]
    scales: tuple[F, ...]
    gram: tuple[tuple[F, ...], ...]
    rhs: tuple[F, ...]
    norm_y2: F
    source_id: str
    source_hash: str
    statistics_hash: str

    @property
    def columns(self) -> int:
        return len(self.scales)


@dataclass(frozen=True)
class Candidate:
    coefficients: tuple[F, ...]
    rss: F
    source_hash: str
    approximate_inverse: tuple[tuple[F, ...], ...] | None = None


def prepare_problem(design: Any, response: Any, *, source_id: str = "") -> PreparedProblem:
    X, y = _matrix(design), _vector(response)
    n, p = len(X), len(X[0])
    if len(y) != n:
        raise NumericInputError("response_length_mismatch")
    if n < p:
        raise NumericInputError("underdetermined_design")
    if p > MAX_COLUMNS:
        raise NumericInputError("column_limit_exceeded")
    if not isinstance(source_id, str):
        raise NumericInputError("invalid_source_identifier")
    scales = tuple(max(abs(row[j]) for row in X) for j in range(p))
    if any(s <= 0 for s in scales):
        raise NumericInputError("zero_design_column")
    Z = tuple(tuple(x / s for x, s in zip(row, scales)) for row in X)
    H = [[F(0) for _ in range(p)] for _ in range(p)]
    for j in range(p):
        for k in range(j, p):
            H[j][k] = H[k][j] = sum((row[j] * row[k] for row in Z), F(0))
    H = tuple(tuple(row) for row in H)
    h = tuple(sum((row[j] * target for row, target in zip(Z, y)), F(0)) for j in range(p))
    c = _dot(y, y)
    _require_positive_definite(H)
    source_hash = _source_hash(X, y, source_id)
    return PreparedProblem(X, y, scales, H, h, c, source_id, source_hash,
                           _statistics_hash(H, h, c, scales, source_hash))


def _check_problem(problem: PreparedProblem) -> None:
    if not isinstance(problem, PreparedProblem):
        raise NumericInputError("unprepared_problem")
    if not isinstance(problem.source_id, str):
        raise NumericInputError("invalid_source_identifier")
    X, y = _matrix(problem.design), _vector(problem.response)
    p = len(X[0])
    if len(y) != len(X) or len(X) < p or p > MAX_COLUMNS:
        raise NumericInputError("invalid_prepared_dimensions")
    scales = _vector(problem.scales)
    H, h, c = _matrix(problem.gram), _vector(problem.rhs), _fraction(problem.norm_y2)
    if len(scales) != p or any(s <= 0 for s in scales) or len(H) != p or len(H[0]) != p or len(h) != p:
        raise NumericInputError("invalid_statistics_dimensions")
    if _source_hash(X, y, problem.source_id) != problem.source_hash:
        raise NumericInputError("source_integrity_failed")
    if _statistics_hash(H, h, c, scales, problem.source_hash) != problem.statistics_hash:
        raise NumericInputError("statistics_integrity_failed")
    # A checksum alone is not an admission proof. Check the sufficient statistics
    # against the actual bound source, even for manually reconstructed objects.
    expected_scales = tuple(max(abs(row[j]) for row in X) for j in range(p))
    if scales != expected_scales:
        raise NumericInputError("statistics_source_mismatch")
    Z = tuple(tuple(x / s for x, s in zip(row, scales)) for row in X)
    if c != _dot(y, y) or any(h[j] != sum((row[j] * target for row, target in zip(Z, y)), F(0))
                               for j in range(p)):
        raise NumericInputError("statistics_source_mismatch")
    for j in range(p):
        for k in range(j, p):
            expected = sum((row[j] * row[k] for row in Z), F(0))
            if H[j][k] != expected or H[k][j] != expected:
                raise NumericInputError("statistics_source_mismatch")
    _require_positive_definite(H)


def _linear_system(matrix: Any, rhs: Any) -> tuple[tuple, tuple]:
    A, b = _matrix(matrix), _vector(rhs)
    if len(A) != len(A[0]) or len(b) != len(A):
        raise NumericInputError("linear_system_dimension_mismatch")
    if len(A) > MAX_COLUMNS:
        raise NumericInputError("column_limit_exceeded")
    return A, b


def reference_fraction_solve(matrix: Any, rhs: Any) -> tuple[F, ...]:
    """Ordinary rational elimination, retained as a separate small oracle."""
    A, b = _linear_system(matrix, rhs)
    p = len(A)
    rows = [list(row) + [target] for row, target in zip(A, b)]
    for k in range(p):
        pivot = next((i for i in range(k, p) if rows[i][k]), None)
        if pivot is None:
            raise NumericInputError("singular_linear_system")
        rows[k], rows[pivot] = rows[pivot], rows[k]
        pivot_value = rows[k][k]
        rows[k] = [v / pivot_value for v in rows[k]]
        for i in range(k + 1, p):
            factor = rows[i][k]
            rows[i] = [a - factor * v for a, v in zip(rows[i], rows[k])]
    answer = [F(0)] * p
    for i in range(p - 1, -1, -1):
        answer[i] = rows[i][-1] - _dot(rows[i][i + 1:p], answer[i + 1:p])
    return tuple(answer)


def bareiss_solve(matrix: Any, rhs: Any) -> tuple[F, ...]:
    """Clear each augmented row's denominators, then fraction-free eliminate."""
    A, b = _linear_system(matrix, rhs)
    p = len(A)
    rows = []
    for row, target in zip(A, b):
        values = tuple(row) + (target,)
        denominator = math.lcm(*(v.denominator for v in values))
        rows.append([v.numerator * (denominator // v.denominator) for v in values])
    previous = 1
    for k in range(p - 1):
        pivot = next((i for i in range(k, p) if rows[i][k]), None)
        if pivot is None:
            raise NumericInputError("singular_linear_system")
        rows[k], rows[pivot] = rows[pivot], rows[k]
        pivot_value = rows[k][k]
        for i in range(k + 1, p):
            for j in range(k + 1, p + 1):
                numerator = pivot_value * rows[i][j] - rows[i][k] * rows[k][j]
                quotient, remainder = divmod(numerator, previous)
                if remainder:
                    raise ArithmeticError("bareiss_nonexact_division")
                rows[i][j] = quotient
            rows[i][k] = 0
        previous = pivot_value
    if not rows[-1][-2]:
        raise NumericInputError("singular_linear_system")
    answer = [F(0)] * p
    for i in range(p - 1, -1, -1):
        if not rows[i][i]:
            raise NumericInputError("singular_linear_system")
        answer[i] = (F(rows[i][-1]) - _dot(rows[i][i + 1:p], answer[i + 1:p])) / rows[i][i]
    return tuple(answer)


def independent_residual(problem: PreparedProblem, coefficients: Any) -> tuple[tuple[F, ...], F, tuple[F, ...]]:
    """Return original-space residuals, RSS, and scaled stationarity residual."""
    beta = _vector(coefficients)
    if len(beta) != problem.columns:
        raise NumericInputError("coefficient_dimension_mismatch")
    residual = tuple(_dot(row, beta) - target for row, target in zip(problem.design, problem.response))
    r = tuple(sum((row[j] * value for row, value in zip(problem.design, residual)), F(0)) / problem.scales[j]
              for j in range(problem.columns))
    return residual, _dot(residual, residual), r


def make_candidate(problem: PreparedProblem, coefficients: Any, *, rss: Any = None,
                   approximate_inverse: Any = None) -> Candidate:
    _check_problem(problem)
    beta = _vector(coefficients)
    _, actual_rss, _ = independent_residual(problem, beta)
    inverse = None if approximate_inverse is None else _matrix(approximate_inverse)
    return Candidate(beta, actual_rss if rss is None else _fraction(rss), problem.source_hash, inverse)


def solve_exact(problem: PreparedProblem, *, method: str = "bareiss") -> Candidate:
    _check_problem(problem)
    if method == "bareiss":
        theta = bareiss_solve(problem.gram, problem.rhs)
    elif method == "reference":
        theta = reference_fraction_solve(problem.gram, problem.rhs)
    else:
        raise NumericInputError("unsupported_exact_method")
    beta = tuple(t / s for t, s in zip(theta, problem.scales))
    _, rss, stationarity = independent_residual(problem, beta)
    if any(stationarity):
        raise ArithmeticError("independent_stationarity_failed")
    return Candidate(beta, rss, problem.source_hash)


def _ratios(beta: tuple, rss: F, claimed_rss: F, r: tuple, scales: tuple, u: tuple) -> tuple[F, F, tuple]:
    coordinate = tuple(v / s for v, s in zip(u, scales))
    error = max(coordinate)
    beta_ratio = error / max(F(1), max(map(abs, beta)) - error)
    gap = sum((bound * abs(value) for bound, value in zip(u, r)), F(0))
    rss_ratio = (abs(claimed_rss - rss) + gap) / max(F(1), rss - gap)
    return beta_ratio, rss_ratio, coordinate


def verify_candidate(problem: PreparedProblem, candidate: Candidate) -> dict:
    """Fail closed with exact, inspectable bounds at the immutable 1e-8 tolerance."""
    result = {"passed": False, "tolerance": str(TOLERANCE), "exact_arithmetic": True,
              "reference_solution_access": False, "neumann_terms": NEUMANN_TERMS}
    try:
        _check_problem(problem)
        if not isinstance(candidate, Candidate):
            raise NumericInputError("invalid_candidate_type")
        if candidate.source_hash != problem.source_hash:
            raise NumericInputError("candidate_source_binding_failed")
        beta, claimed_rss = _vector(candidate.coefficients), _fraction(candidate.rss)
        if len(beta) != problem.columns:
            raise NumericInputError("coefficient_dimension_mismatch")
        if claimed_rss < 0:
            raise NumericInputError("negative_claimed_rss")
        _, rss, r = independent_residual(problem, beta)
        result["independent_rss"] = str(rss)
        if not any(r):
            passed = claimed_rss == rss
            return {**result, "passed": passed,
                    "reason": "exact_stationarity" if passed else "exact_rss_mismatch",
                    "coordinate_error_bounds": ["0"] * problem.columns,
                    "coefficient_relative_bound": "0", "rss_relative_bound": str(abs(claimed_rss - rss) / max(F(1), rss))}
        if candidate.approximate_inverse is None:
            raise NumericInputError("approximate_inverse_required")
        B = _matrix(candidate.approximate_inverse)
        p = problem.columns
        if len(B) != p or len(B[0]) != p:
            raise NumericInputError("inverse_dimension_mismatch")
        H = problem.gram
        M = tuple(tuple(abs(F(i == j) - sum((B[i][k] * H[k][j] for k in range(p)), F(0)))
                        for j in range(p)) for i in range(p))
        q = max(sum(row, F(0)) for row in M)
        result["q"] = str(q)
        if q >= 1:
            raise NumericInputError("contraction_not_verified")
        v = tuple(abs(value) for value in _matvec(B, r))
        scalar = max(v) / (1 - q)
        term, total = v, v
        for _ in range(1, NEUMANN_TERMS):
            term = _matvec(M, term)
            total = tuple(a + b for a, b in zip(total, term))
        next_term = _matvec(M, term)
        tail = max(next_term) / (1 - q)
        u = tuple(min(scalar, value + tail) for value in total)
        supersolution = all(a + b <= bound for a, b, bound in zip(v, _matvec(M, u), u))
        if not supersolution:
            raise ArithmeticError("componentwise_supersolution_failed")
        beta_bound, rss_bound, coordinate = _ratios(beta, rss, claimed_rss, r, problem.scales, u)
        old_beta, old_rss, old_coordinate = _ratios(beta, rss, claimed_rss, r, problem.scales, (scalar,) * p)
        # Preserve the actual previous certificate's coarser objective-gap rule,
        # not only the scalar-coordinate rule used by _ratios above.
        old_gap = p * scalar * max(map(abs, r))
        old_rss = (abs(claimed_rss - rss) + old_gap) / max(F(1), rss - old_gap)
        result.update({"coordinate_error_bounds": list(map(str, coordinate)),
                       "scaled_coordinate_error_bounds": list(map(str, u)),
                       "old_scalar_coordinate_bounds": list(map(str, old_coordinate)),
                       "coefficient_relative_bound": str(beta_bound), "rss_relative_bound": str(rss_bound),
                       "old_scalar_coefficient_relative_bound": str(old_beta),
                       "old_scalar_rss_relative_bound": str(old_rss),
                       "supersolution_verified": True, "neumann_tail_bound": str(tail)})
        result["passed"] = beta_bound <= TOLERANCE and rss_bound <= TOLERANCE
        result["reason"] = ("bounded" if result["passed"] else
                            "coefficient_and_rss_bounds_too_wide" if beta_bound > TOLERANCE and rss_bound > TOLERANCE else
                            "coefficient_bound_too_wide" if beta_bound > TOLERANCE else "rss_bound_too_wide")
    except (NumericInputError, ArithmeticError, TypeError, AttributeError) as exc:
        result["reason"] = str(exc)
    return result
