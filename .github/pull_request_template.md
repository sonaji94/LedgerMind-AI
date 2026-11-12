## Description
Please include a summary of the change and which issue it addresses.

Fixes #(issue)

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance / Refactoring

## Accounting Invariant & Financial Integrity
- [ ] Does this PR touch monetary balance logic or journal entries?
- [ ] Confirmed that $\Sigma\text{Debits} == \Sigma\text{Credits}$ holds with 0 drift.
- [ ] Uses integer minor units (`amount_minor_units`) for all financial amounts (no floating-point money).

## Testing & Verification
- [ ] Unit tests added / updated and passing (`pytest tests/unit/`)
- [ ] Integration tests passing (`pytest tests/integration/`)
- [ ] API endpoint tests passing (`pytest tests/api/`)
- [ ] Linter passing with zero warnings (`ruff check .`)
