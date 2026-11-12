---
name: Bug Report
about: Create a report to help us improve LedgerMind AI
title: '[BUG] '
labels: 'bug, triage'
assignees: ''
---

**Describe the Bug**
A clear and concise description of what the bug is.

**Affected Module**
- [ ] Ledger Engine (`app/ledger`)
- [ ] Transfer / Transaction API (`app/transactions`)
- [ ] Fraud Detection & Inference Service (`app/fraud`)
- [ ] Settlement Analytics (`app/analytics`)
- [ ] AI Assistant / RAG Agent (`app/ai_assistant`)
- [ ] Streamlit Cockpit (`app/dashboard.py`)
- [ ] Core / Security / Auth (`app/core`, `app/auth`)

**Accounting Invariant Check**
Did this issue cause or expose an accounting drift?
- Total Debits:
- Total Credits:
- Drift ($\Delta$):

**To Reproduce**
Steps to reproduce the behavior:
1. Call endpoint `...` or execute script `...`
2. With payload `...`
3. See error

**Expected Behavior**
A clear description of what you expected to happen (e.g. Invariant $\Sigma\text{Debits} == \Sigma\text{Credits}$ holds, exact 400 validation returned).

**Environment:**
- OS: [e.g. Windows 11, Ubuntu 22.04]
- Python Version: [e.g. 3.12, 3.15]
- Database: [e.g. SQLite, PostgreSQL 16]

**Additional Context**
Add any other context or terminal tracebacks about the problem here.
