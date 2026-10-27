# Research safety and local operation

This is a local, single-user prototype, not a hardened network service. Bind
to loopback only; do not expose it through a tunnel, public proxy or shared
hospital network. It is not a clinical system or an authentication service.

The release accepts generated inputs only. It must not load patient data,
labels, protected cohort records or model weights. Natural-language routing
does not execute shell commands or arbitrary code. External model plugins
and patient execution are disabled. There is no autonomous diagnosis,
treatment recommendation or background training.

Runtime state is stored separately from source. Audit hashes detect accidental
modification against stored receipts, but an attacker who controls both the
database and receipts can forge them. A hash chain is not a digital signature,
trusted time source or substitute for institutional provenance review.

Never commit runtime databases, personal records, signed download URLs,
credentials or weights. Report suspected source-code vulnerabilities to the
repository owner without attaching protected data or credentials.
