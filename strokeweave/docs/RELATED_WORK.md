# Related work and scientific boundaries

StrokeWeave is a research-software release, not a new visual foundation model or an autonomous clinical system. Its numerical utilities illustrate auditable verification and fallback. They do not establish a new theorem, an improvement on the frozen NIST experiment, or a patient-level stroke benefit.

## Verified least squares and componentwise bounds

Rump's **Improved componentwise verified error bounds for least squares problems and underdetermined linear systems** develops rigorous componentwise enclosures, including rounding error, using QR-related preconditioning and residual information. It also discusses residual iteration and staggered correction. These are established techniques that can justify improvements to a verification component, not a StrokeWeave novelty claim. The paper's assumptions and arithmetic safeguards must be implemented before citing its guarantees for a particular algorithm.

- Journal reference: S. M. Rump, *Numerical Algorithms* **66**, 309–322 (2014), [doi:10.1007/s11075-013-9735-6](https://doi.org/10.1007/s11075-013-9735-6).
- [Author manuscript](https://www.tuhh.de/ti3/paper/rump/Ru13a.pdf).

For an exact linear system, a useful elementary specialization is to write

\[
r=H\widehat\theta-h,\qquad C=I-BH,\qquad z=Br.
\]

If an exactly verified induced-norm bound satisfies \(q=\lVert C\rVert_\infty<1\), then \(H\) is nonsingular and the error obeys

\[
\widehat\theta-\theta^\ast=(I-C)^{-1}z.
\]

A scalar enclosure is \(\lVert z\rVert_\infty/(1-q)\). A componentwise enclosure can retain more structure. With \(v_0=|z|\), \(v_{i+1}=|C|v_i\), and a preselected integer \(K\), one valid enclosure is

\[
u_K=\sum_{i=0}^{K-1}v_i+
\frac{\lVert v_K\rVert_\infty}{1-q}\boldsymbol 1.
\]

Alternatively, a nonnegative vector satisfying the exactly checked inequality
\(|z|+|C|u\le u\), together with the contraction condition, is a valid supersolution. These are applications of the standard Neumann-series/contraction argument, not newly discovered mechanisms. A floating-point estimate of such a vector is not itself a proof.

For a full-column-rank least-squares design with exact \(H=Z^\top Z\), the objective excess is \((\widehat\theta-\theta^\ast)^\top r\). Therefore a valid componentwise error enclosure gives an objective-gap bound \(\sum_j u_j|r_j|\). This can be tighter than multiplying a scalar error radius by dimension and the maximum residual. Linear-system invertibility alone does not justify applying this least-squares interpretation to an arbitrary matrix.

## Backward error is not a forward-accuracy certificate

The official LAPACK Users' Guide distinguishes small backward error, condition-dependent forward error, rank handling, and least-squares residuals. It also describes routes to componentwise error estimates. Consequently, a small residual or successful solver return must not be treated as sufficient proof that coefficients meet a fixed forward-accuracy tolerance. Approximate condition estimates are useful diagnostics but are not automatically rigorous enclosures.

- [LAPACK: Error Bounds for Linear Least Squares Problems](https://netlib.org/lapack/lug/node82.html).
- [LAPACK: Further Details](https://netlib.org/lapack/lug/node83.html).

## Fraction-free exact arithmetic

Bareiss introduced integer-preserving elimination based on determinant identities. Clearing rational denominators and using fraction-free elimination can avoid repeated rational normalization during elimination. This does not eliminate integer growth, guarantee faster execution for every small matrix, or remove the need to check pivots, exact divisibility, and the final residual. It is an established exact-arithmetic baseline, not an agent innovation.

- E. H. Bareiss, **Sylvester's identity and multistep integer-preserving Gaussian elimination**, *Mathematics of Computation* **22**, 565–578 (1968), [doi:10.1090/S0025-5718-1968-0226829-0](https://doi.org/10.1090/S0025-5718-1968-0226829-0).

## Verification cost and fair comparison

Rump and Ogita give an established example of efficient validated solution for symmetric positive-definite linear systems. It supports investigating efficient certificates, but its result is not a universal runtime guarantee for this Python implementation or every task family.

- S. M. Rump and T. Ogita, **Super-fast validated solution of linear systems**, *Journal of Computational and Applied Mathematics* **199**, 199–206 (2007), [institutional publication record and DOI](https://tore.tuhh.de/entities/publication/5b142080-1d1d-4f5b-8b95-1f589deb287e).

Lazy imports, shared sufficient statistics, symmetric matrix construction, residual reuse, and better exact elimination can improve a common tool. Every comparator must receive these improvements. Faster execution of the same route is not evidence that an adaptive controller chooses better actions.

Cold-start and warm-worker timing should be reported separately. Complete-task cost includes preparation, approximate computation, certificate construction, failed verification, fallback, orchestration, and relevant startup cost. Timing repetitions do not increase the number of independent tasks. The frozen ratio of approximately 0.983 must remain a negative result against its prespecified 0.8 target; release self-tests cannot retroactively change it.

## Naming and release scope

**StrokeWeave** is a descriptive working name for auditable stroke-imaging research workflows. Exact-phrase web and GitHub-oriented searches on 27 October 2026 did not return an obvious identical project. Search coverage is incomplete: this is not trademark clearance, a guarantee of uniqueness, or a claim of clinical authorization. TACTIC remains the historical name of the control research and evidence lineage.

Automated independent-code checks mean that an independently written implementation or oracle checked the software. They are **not** institutional independent review, ethics approval, patient-data admission, or validation of clinical effectiveness. Synthetic tests must be labeled synthetic. Frozen patient roles, protected labels, data licenses, historical exclusions, and prior negative results remain unchanged.
