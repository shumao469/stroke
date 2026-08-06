# Security policy

## Reporting a vulnerability

Do not open a public issue containing credentials, patient identifiers, exploitable details, or sensitive clinical data. Contact the repository owner privately through an institutional channel.

## Credential handling

- Never commit personal access tokens or `.streamlit/secrets.toml`.
- Use a fine-grained GitHub token limited to the target repository.
- Revoke or rotate a token immediately if it appears in a commit, log, screenshot, or uploaded ZIP.

## Clinical data

This is a public repository. Upload only properly de-identified content that is permitted by ethics approval, consent, data-use agreements, institutional policy, and applicable law.
