"""Fresh exact fixtures only; no exposed NIST or patient inputs are imported."""

from dataclasses import replace
from fractions import Fraction as F
import unittest

from strokeweave.numerics import (
    Candidate, NEUMANN_TERMS, TOLERANCE, NumericInputError, bareiss_solve,
    independent_residual, make_candidate, prepare_problem,
    reference_fraction_solve, solve_exact, verify_candidate,
)


class ExactSystemTests(unittest.TestCase):
    def test_fixed_choices(self):
        self.assertEqual(NEUMANN_TERMS, 3)
        self.assertEqual(TOLERANCE, F(1, 100_000_000))

    def test_bareiss_and_reference_fresh_systems(self):
        systems = [
            ([[7]], [3]),
            ([[4, 1], [1, 3]], [1, 2]),
            ([["3/2", "1/7"], ["1/7", "8/3"]], ["2/5", "-7/13"]),
            ([[6, 2, 1], [2, 5, 2], [1, 2, 4]], ["1/3", 7, -2]),
            ([[0, "1/3", 1], [2, "4/5", 0], [1, 0, "7/8"]], [1, -1, 3]),
            ([[0, 2], [3, 4]], [5, 6]),
        ]
        for A, b in systems:
            with self.subTest(A=A):
                actual = bareiss_solve(A, b)
                self.assertEqual(actual, reference_fraction_solve(A, b))
                for row, rhs in zip(A, b):
                    self.assertEqual(sum((F(x) * y for x, y in zip(row, actual)), F(0)), F(rhs))

    def test_singular_systems_rejected(self):
        for solver in (bareiss_solve, reference_fraction_solve):
            with self.assertRaisesRegex(NumericInputError, "singular"):
                solver([[1, 2], [2, 4]], [1, 2])

    def test_exact_fit_independent_stationarity(self):
        X, y = [[1, -2], [1, 0], [1, 3], [1, 4]], [0, 3, 8, 11]
        problem = prepare_problem(X, y, source_id="new-hand-fit")
        a, b = solve_exact(problem), solve_exact(problem, method="reference")
        self.assertEqual(a, b)
        residual, rss, scaled_stationarity = independent_residual(problem, a.coefficients)
        self.assertFalse(any(scaled_stationarity))
        self.assertEqual(rss, sum((r * r for r in residual), F(0)))
        for column in zip(*X):
            self.assertEqual(sum((F(v) * r for v, r in zip(column, residual)), F(0)), 0)
        proof = verify_candidate(problem, a)
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["reason"], "exact_stationarity")

    def test_symmetric_gram_and_source_binding(self):
        p = prepare_problem([["1/3", 2], ["4/3", -1], ["7/3", 5]], [1, 2, 9], source_id="a")
        self.assertEqual(p.gram, tuple(zip(*p.gram)))
        same = prepare_problem([[F(1, 3), 2], [F(4, 3), -1], [F(7, 3), 5]], [1, 2, 9], source_id="a")
        other = prepare_problem(p.design, p.response, source_id="b")
        self.assertEqual(p.source_hash, same.source_hash)
        self.assertNotEqual(p.source_hash, other.source_hash)
        self.assertEqual(verify_candidate(other, solve_exact(p))["reason"], "candidate_source_binding_failed")

    def test_problem_integrity_and_derived_stats(self):
        p = prepare_problem([[1, 0], [0, 1]], [1, 2])
        candidate = solve_exact(p)
        corrupt = replace(p, gram=((F(2), F(0)), (F(0), F(1))))
        self.assertEqual(verify_candidate(corrupt, candidate)["reason"], "statistics_integrity_failed")
        corrupt = replace(p, response=(F(2), F(2)))
        self.assertEqual(verify_candidate(corrupt, candidate)["reason"], "source_integrity_failed")

    def test_bad_inputs_fail_explicitly(self):
        cases = [([], [], "empty_matrix"),
                 ([[1, 2], [3]], [1, 2], "ragged_matrix"),
                 ([[1]], [], "response_length_mismatch"),
                 ([[1, 2]], [1], "underdetermined_design"),
                 ([[1, 0], [2, 0]], [1, 2], "zero_design_column"),
                 ([[1, 2], [2, 4]], [1, 2], "full_column_rank"),
                 ([[True]], [1], "boolean_number"),
                 ([[float("inf")]], [1], "nonfinite"),
                 ([[float("nan")]], [1], "nonfinite"),
                 ([[1]], ["NaN"], "nonfinite"),
                 ([["1/0"]], [1], "nonfinite"),
                 ([[1]], [False], "boolean_number"),
                 ([["1e1000000000"]], [1], "exponent_limit"),
                 ([[1 << 20000]], [1], "bit_limit")]
        for X, y, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(NumericInputError, reason):
                prepare_problem(X, y)

    def test_linear_dimension_errors(self):
        for solver in (bareiss_solve, reference_fraction_solve):
            with self.assertRaisesRegex(NumericInputError, "dimension"):
                solver([[1, 2]], [1])
            with self.assertRaisesRegex(NumericInputError, "dimension"):
                solver([[1]], [1, 2])


class ComponentwiseCertificateTests(unittest.TestCase):
    def setUp(self):
        # Scaled Gram is identity, while original coefficient scales differ by 10^6.
        self.problem = prepare_problem([[1, 0], [0, 1_000_000]], [1, 1], source_id="new-scales")
        self.truth = (F(1), F(1, 1_000_000))
        self.inverse = [[F(9, 10), 0], [0, F(1, 2)]]

    def test_componentwise_bound_encloses_and_tightens(self):
        beta = (self.truth[0] + F(1, 10**9), self.truth[1] + F(1, 10**15))
        candidate = make_candidate(self.problem, beta, approximate_inverse=self.inverse)
        proof = verify_candidate(self.problem, candidate)
        self.assertTrue(proof["passed"], proof)
        self.assertTrue(proof["supersolution_verified"])
        bound = tuple(map(F, proof["coordinate_error_bounds"]))
        old = tuple(map(F, proof["old_scalar_coordinate_bounds"]))
        for estimate, true, u, previous in zip(beta, self.truth, bound, old):
            self.assertLessEqual(abs(estimate - true), u)
            self.assertLessEqual(u, previous)
        self.assertTrue(any(a < b for a, b in zip(bound, old)))
        self.assertLessEqual(F(proof["coefficient_relative_bound"]), F(proof["old_scalar_coefficient_relative_bound"]))
        self.assertLessEqual(F(proof["rss_relative_bound"]), F(proof["old_scalar_rss_relative_bound"]))

    def test_inaccurate_coefficients_are_rejected(self):
        for error in (F(1, 10**5), -F(1, 10**5), F(100)):
            beta = (self.truth[0] + error, self.truth[1])
            candidate = make_candidate(self.problem, beta, approximate_inverse=self.inverse)
            proof = verify_candidate(self.problem, candidate)
            self.assertFalse(proof["passed"], proof)
            self.assertIn("coefficient", proof["reason"])

    def test_wrong_rss_is_not_hidden_by_good_coefficients(self):
        exact = solve_exact(self.problem)
        proof = verify_candidate(self.problem, replace(exact, rss=F(1)))
        self.assertFalse(proof["passed"])
        self.assertEqual(proof["reason"], "exact_rss_mismatch")
        beta = (self.truth[0] + F(1, 10**10), self.truth[1])
        candidate = make_candidate(self.problem, beta, rss=1, approximate_inverse=self.inverse)
        self.assertFalse(verify_candidate(self.problem, candidate)["passed"])

    def test_identity_preconditioner_exact_error_bound(self):
        beta = (self.truth[0] + F(1, 10**10), self.truth[1] - F(1, 10**16))
        proof = verify_candidate(self.problem, make_candidate(self.problem, beta, approximate_inverse=[[1, 0], [0, 1]]))
        self.assertTrue(proof["passed"])
        self.assertEqual(F(proof["q"]), 0)
        self.assertEqual(tuple(map(F, proof["coordinate_error_bounds"])),
                         tuple(abs(a - b) for a, b in zip(beta, self.truth)))

    def test_anisotropic_fixture_reduces_conservative_refusal_without_relaxing_tolerance(self):
        # A new analytic implementation test, not a replay of the four old refusals.
        beta = (self.truth[0] + F(1, 10**9), self.truth[1] + F(1, 10**12))
        proof = verify_candidate(self.problem, make_candidate(
            self.problem, beta, approximate_inverse=[[F(1, 2), 0], [0, 1]]))
        self.assertTrue(proof["passed"], proof)
        self.assertLessEqual(F(proof["coefficient_relative_bound"]), TOLERANCE)
        self.assertGreater(F(proof["old_scalar_coefficient_relative_bound"]), TOLERANCE)
        for bound, actual, true in zip(map(F, proof["coordinate_error_bounds"]), beta, self.truth):
            self.assertGreaterEqual(bound, abs(actual - true))

    def test_missing_inverse_and_noncontraction_rejected(self):
        beta = (self.truth[0] + F(1, 10**9), self.truth[1])
        c = make_candidate(self.problem, beta)
        self.assertEqual(verify_candidate(self.problem, c)["reason"], "approximate_inverse_required")
        bad = replace(c, approximate_inverse=((F(0), F(0)), (F(0), F(0))))
        self.assertEqual(verify_candidate(self.problem, bad)["reason"], "contraction_not_verified")

    def test_bad_candidate_inputs_fail_closed(self):
        good = solve_exact(self.problem)
        bad_cases = [replace(good, coefficients=(F(1),)),
                     replace(good, coefficients=(float("nan"), F(0))),
                     replace(good, rss=float("inf")),
                     replace(good, rss=F(-1)),
                     replace(good, source_hash="wrong"),
                     {"coefficients": [1, 2]}]
        for bad in bad_cases:
            with self.subTest(candidate=bad):
                self.assertFalse(verify_candidate(self.problem, bad)["passed"])

    def test_dense_bound_with_independent_two_by_two_oracle(self):
        p = prepare_problem([[1, 0], [1, 1], [0, 1]], [1, 4, 2])
        # Independent 2x2 unscaled normal equations: [[2,1],[1,2]] b=[5,6].
        truth = (F(4, 3), F(7, 3))
        B = [[F(3, 5), F(-3, 10)], [F(-3, 10), F(3, 5)]]
        for e1, e2 in ((F(1, 10**10), F(-2, 10**10)),
                       (F(1, 10**4), F(1, 10**5)),
                       (-F(1, 10**12), F(0))):
            beta = tuple(a + b for a, b in zip(truth, (e1, e2)))
            c = make_candidate(p, beta, approximate_inverse=B)
            proof = verify_candidate(p, c)
            self.assertEqual(proof.get("supersolution_verified"), True, proof)
            bounds = tuple(map(F, proof["coordinate_error_bounds"]))
            for e, u in zip((e1, e2), bounds):
                self.assertLessEqual(abs(e), u)
            true_relative = max(abs(e1), abs(e2)) / max(F(1), max(map(abs, truth)))
            if proof["passed"]:
                self.assertLessEqual(true_relative, TOLERANCE)
                _, true_rss, _ = independent_residual(p, truth)
                self.assertLessEqual(abs(c.rss - true_rss) / max(F(1), true_rss), TOLERANCE)


if __name__ == "__main__":
    unittest.main()
