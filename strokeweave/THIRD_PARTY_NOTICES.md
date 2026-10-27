# Methods and dependencies

The portable core uses the Python standard library. It includes no copied
third-party solver source, patient files, restricted datasets or model weights.
The numerical methods are implementations of established ideas, including
fraction-free elimination and verified componentwise error bounds. Their
primary references and mathematical preconditions are listed in
`docs/RELATED_WORK.md` and `docs/NUMERICS_AUDIT.md`.

DeepISLES, SynthStroke and Prefect are not bundled or required by this core.
Prior private experiments using them do not confer redistribution rights or
make this package an implementation of those tools. A future adapter must
record the exact model/source version, applicable licence, training exposure,
modalities and execution permissions before activation. In particular,
ISLES-exposed models cannot be called unseen on the corresponding data.

The aggregate historical NIST results in `docs/HISTORICAL_RESULTS.md` are not
a redistribution of the NIST datasets. See the
[official NIST reference-data description](https://www.nist.gov/itl/sed/statistical-reference-datasets/strd-background-information/linear-regression)
for the original sources, which are excluded from this release.
