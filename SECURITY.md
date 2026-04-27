# Security policy

## Supported versions

Pre-launch baseline. Only `0.1.x` on `main` receives fixes; older
in-development versions are unsupported.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1.0 | No        |

## Reporting a vulnerability

**Please do not file a public issue for security bugs.**

Use GitHub's private vulnerability reporting:

[**Report a vulnerability**](https://github.com/Uranid/mnem/security/advisories/new)

In your report, include:

- The affected version, commit hash, or release tag
- A minimal reproduction (steps, code snippet, corpus bytes)
- The observed impact (panic, corruption, signature forgery, memory
  safety, etc.)
- Whether the finding has been disclosed elsewhere

## Response timeline

- **Acknowledgement**: within 2 business days.
- **Triage**: within 5 business days. You'll hear whether the issue is
  in scope, what severity we assess it at, and a target fix window.
- **Fix**: timeline depends on severity. Critical issues (remote
  corruption, signature forgery, use-after-free in safe-only code)
  target a patch release within 14 days. Lower-severity issues land
  on the normal release schedule.
- **Disclosure**: coordinated. A public advisory is drafted alongside
  the patch; we publish the advisory no later than 90 days after the
  initial report, sooner if a fix is already shipped.

## Scope

In scope:

- `mnem-core`, `mnem-backend-redb`, and any other first-party crate in
  this workspace.
- Published documentation where a doc bug could lead to an insecure
  deployment.

Out of scope:

- Third-party backends or bindings (report to their own maintainers).
- General Rust / operating-system security issues not specific to mnem.
- Denial-of-service from obviously adversarial inputs when the caller
  is expected to pre-validate (documented boundaries).

## Hardening commitments

`mnem-core` is `#![forbid(unsafe_code)]`. Every parser MUST return `Err`
on arbitrary bytes without panicking; this is covered by a
cross-platform proptest suite plus a coverage-guided fuzz harness
(see [`fuzz/`](fuzz/) in the repo root) and is a release blocker.

Signing uses Ed25519 via `ed25519-dalek`. Key revocation is first-class.

## Credit

Reporters are credited in the advisory unless they request anonymity.
No bug bounty at this time; responsible disclosures appreciated.
