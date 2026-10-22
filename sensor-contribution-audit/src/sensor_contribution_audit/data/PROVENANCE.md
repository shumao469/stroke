# Analytical reference provenance

`figure1a_reference.csv` retains the prior equation-derived Figure 1a timing
illustration grid dated 22 October 2026. It is not participant data or an
independent experimental validation dataset. Every value comes from
`exp(-0.5 * (2*pi*f*sigma_ms/1000)^2)`.

The table covers all pairs of integer frequency 1-40 Hz and integer residual
relative timing standard deviation 0-20 ms, for 840 values. Coordinates and
values retain their original column names. A SHA-256 digest accompanies the
table. Repository tests regenerate these values from the closed form and use
absolute tolerance `1e-14` to accommodate floating-point evaluation order.

The retained table contains no person identifiers, local-machine paths, clinical
labels or author contact details. The package contains no source articles,
manuscript documents or internal cohort results. The separate `reported/`
directory contains attributed aggregate transcriptions from two articles,
with their own source map and licensing scope.
