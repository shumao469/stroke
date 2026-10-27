"""Independent exact-oracle checks on newly generated non-patient inputs.

The oracle uses determinant expansion and Cramer's rule, not either production
elimination implementation. These are software checks, not institutional review,
patient admission, old-NIST re-evaluation, or a performance comparison.
"""

from fractions import Fraction as F
from itertools import product
import unittest

from strokeweave.numerics import (
    Candidate,
    NumericInputError,
    TOLERANCE,
    bareiss_solve,
    make_candidate,
    prepare_problem,
    solve_exact,
    verify_candidate,
)


def determinant(rows):
    """Laplace expansion: deliberately small and unlike the production solver."""
    if not rows:
        return F(1)
    if len(rows) == 1:
        return F(rows[0][0])
    total = F(0)
    for j, value in enumerate(rows[0]):
        minor = [list(row[:j]) + list(row[j + 1:]) for row in rows[1:]]
        total += (-1) ** j * F(value) * determinant(minor)
    return total


def cramer(rows, target):
    det = determinant(rows)
    if not det:
        raise ValueError("singular oracle input")
    answer = []
    for j in range(len(rows)):
        replaced = [list(map(F, row)) for row in rows]
        for i, value in enumerate(target):
            replaced[i][j] = F(value)
        answer.append(determinant(replaced) / det)
    return tuple(answer)


def inverse(rows):
    p = len(rows)
    columns = [cramer(rows, [F(i == j) for i in range(p)]) for j in range(p)]
    return tuple(tuple(columns[j][i] for j in range(p)) for i in range(p))


def least_squares_oracle(design, response):
    p = len(design[0])
    gram = [[sum((F(row[j]) * F(row[k]) for row in design), F(0))
             for k in range(p)] for j in range(p)]
    rhs = [sum((F(row[j]) * F(y) for row, y in zip(design, response)), F(0))
           for j in range(p)]
    beta = cramer(gram, rhs)
    residual = [sum((F(x) * b for x, b in zip(row, beta)), F(0)) - F(y)
                for row, y in zip(design, response)]
    return beta, sum((r * r for r in residual), F(0))


def generated_design(p, fixture):
    # The identity block makes full column rank analytic, without trusting a
    # floating-point rank estimate or selecting fixtures based on outcomes.
    rows = [[F(int(i == j) * (j + 2), fixture + 2) for j in range(p)]
            for i in range(p)]
    rows += [[F(((i + 2) * (j + 3) + fixture) % 7 - 3, j + 2)
              for j in range(p)] for i in range(3)]
    target = [F(((i + 1) * (fixture + 2)) % 11 - 5, i + 2)
              for i in range(len(rows))]
    return rows, target


class IndependentNumericsChecks(unittest.TestCase):
    def test_bareiss_all_small_two_by_two_nonsingular_inputs(self):
        checked = 0
        for entries in product((-1, 0, 1), repeat=4):
            rows = [list(map(F, entries[:2])), list(map(F, entries[2:]))]
            if not determinant(rows):
                continue
            target = [F(1, 2), F(-2, 3)]
            with self.subTest(entries=entries):
                self.assertEqual(bareiss_solve(rows, target), cramer(rows, target))
            checked += 1
        self.assertEqual(checked, 48)

    def test_bareiss_row_denominators_and_later_pivot_swap(self):
        cases = [
            ([[0, F(1, 3), F(2, 5)], [F(1, 7), F(2, 11), 0],
              [F(2, 13), 0, F(-1, 17)]], [F(3, 19), -1, F(2, 23)]),
            ([[1, 1, 0], [1, 1, 1], [0, 1, 1]], [F(2, 3), F(1, 5), F(-7, 11)]),
        ]
        for rows, target in cases:
            with self.subTest(rows=rows):
                self.assertEqual(bareiss_solve(rows, target), cramer(rows, target))

    def test_exact_least_squares_matches_original_space_cramer_oracle(self):
        for p in range(1, 5):
            for fixture in range(4):
                rows, target = generated_design(p, fixture)
                oracle_beta, oracle_rss = least_squares_oracle(rows, target)
                problem = prepare_problem(rows, target, source_id=f"independent-{p}-{fixture}")
                solved = solve_exact(problem)
                with self.subTest(p=p, fixture=fixture):
                    self.assertEqual(solved.coefficients, oracle_beta)
                    self.assertEqual(solved.rss, oracle_rss)
                    self.assertTrue(verify_candidate(problem, solved)["passed"])

    def test_componentwise_bounds_cover_true_errors_without_solver_access(self):
        comparisons = 0
        for p in range(1, 5):
            for fixture in range(4):
                rows, target = generated_design(p, fixture)
                oracle_beta, oracle_rss = least_squares_oracle(rows, target)
                problem = prepare_problem(rows, target, source_id=f"bounds-{p}-{fixture}")
                exact_inverse = inverse(problem.gram)
                for attenuation in (F(0), F(1, 4), F(3, 4)):
                    B = tuple(tuple((1 - attenuation) * x for x in row)
                              for row in exact_inverse)
                    beta = tuple(x + F((-1) ** j * (j + 1), 10 ** 12)
                                 for j, x in enumerate(oracle_beta))
                    candidate = make_candidate(problem, beta, approximate_inverse=B)
                    result = verify_candidate(problem, candidate)
                    with self.subTest(p=p, fixture=fixture, attenuation=attenuation):
                        self.assertEqual(F(result["q"]), attenuation)
                        self.assertTrue(result["supersolution_verified"])
                        self.assertFalse(result["reference_solution_access"])
                        coordinate = tuple(map(F, result["coordinate_error_bounds"]))
                        actual_errors = tuple(abs(a - b) for a, b in zip(beta, oracle_beta))
                        self.assertTrue(all(a <= b for a, b in zip(actual_errors, coordinate)))
                        true_beta_ratio = max(actual_errors) / max(F(1), max(map(abs, oracle_beta)))
                        true_rss_ratio = abs(candidate.rss - oracle_rss) / max(F(1), oracle_rss)
                        self.assertLessEqual(true_beta_ratio, F(result["coefficient_relative_bound"]))
                        self.assertLessEqual(true_rss_ratio, F(result["rss_relative_bound"]))
                        if result["passed"]:
                            self.assertLessEqual(true_beta_ratio, TOLERANCE)
                            self.assertLessEqual(true_rss_ratio, TOLERANCE)
                    comparisons += 1
        self.assertEqual(comparisons, 48)

    def test_deliberately_inaccurate_candidates_cannot_be_accepted(self):
        rows, target = generated_design(3, 2)
        beta, rss = least_squares_oracle(rows, target)
        problem = prepare_problem(rows, target, source_id="independent-bad-candidate")
        B = inverse(problem.gram)
        wrong_beta = (beta[0] + 1, beta[1], beta[2])
        self.assertFalse(verify_candidate(problem, make_candidate(
            problem, wrong_beta, approximate_inverse=B))["passed"])
        self.assertFalse(verify_candidate(problem, make_candidate(
            problem, beta, rss=rss + 1, approximate_inverse=B))["passed"])
        unbound = Candidate(beta, rss, "0" * 64, B)
        self.assertFalse(verify_candidate(problem, unbound)["passed"])

    def test_nondiagonal_signed_contraction_bound_with_independent_oracle(self):
        for p in range(2, 5):
            for fixture in range(4):
                rows, target = generated_design(p, fixture)
                truth, true_rss = least_squares_oracle(rows, target)
                problem = prepare_problem(rows, target, source_id=f"signed-{p}-{fixture}")
                H_inverse = inverse(problem.gram)
                C = [[F(1, 5) if i == j else F((-1) ** (i + j + fixture), 10 * (p + 1))
                      for j in range(p)] for i in range(p)]
                # Construct B=(I-C)H^-1, so the true contraction is known without
                # invoking any production solver or certificate helper.
                B = [[sum(((F(i == k) - C[i][k]) * H_inverse[k][j]
                           for k in range(p)), F(0)) for j in range(p)] for i in range(p)]
                beta = tuple(x + F((-1) ** (j + fixture) * (j + 1), 10 ** 11)
                             for j, x in enumerate(truth))
                candidate = make_candidate(problem, beta, approximate_inverse=B)
                result = verify_candidate(problem, candidate)
                with self.subTest(p=p, fixture=fixture):
                    self.assertEqual(F(result["q"]), max(sum(map(abs, row), F(0)) for row in C))
                    self.assertTrue(result["supersolution_verified"])
                    bounds = tuple(map(F, result["coordinate_error_bounds"]))
                    self.assertTrue(all(abs(a - b) <= u for a, b, u in zip(beta, truth, bounds)))
                    self.assertLessEqual(abs(candidate.rss - true_rss) / max(F(1), true_rss),
                                         F(result["rss_relative_bound"]))

    def test_no_contraction_or_rank_cannot_be_certified(self):
        problem = prepare_problem([[1, 0], [0, 1], [1, 1]], [1, 2, 2])
        candidate = make_candidate(problem, [0, 0], approximate_inverse=[[0, 0], [0, 0]])
        result = verify_candidate(problem, candidate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "contraction_not_verified")
        with self.assertRaises(NumericInputError):
            prepare_problem([[1, 2], [2, 4], [3, 6]], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
