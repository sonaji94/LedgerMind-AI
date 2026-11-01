# LedgerMind AI — Security Architecture & Threat Model

This document outlines the security architecture, assumptions, threat mitigations, and compliance boundaries for **LedgerMind AI**.

---

## 1. System Security Principles

1. **Defense in Depth**: Security controls are applied at the network, gateway, service, domain, and database layers.
2. **Authority vs. Advisory Separation**:
   - The **Financial Ledger** is deterministic, authoritative, and strictly rule-based.
   - The **Machine Learning Model** is probabilistic and strictly advisory. Predictions cannot alter financial balances, authorize account freezes, or adjudicate fraud guilt without human analyst review.
3. **Immutability of Historical State**:
   - `journal_transactions`, `journal_entries`, and `domain_events` are append-only.
   - Database user roles in production are granted `INSERT` and `SELECT` on financial journals, but `UPDATE` and `DELETE` are revoked.

---

## 2. STRIDE Threat Analysis & Mitigations

| Category | Threat Description | Architectural Mitigation |
|:---|:---|:---|
| **Spoofing** | Untrusted client impersonates an account owner or passes arbitrary account IDs. | Server-side bearer authentication (JWT) and ownership enforcement. The server extracts the caller identity from the token rather than trusting request bodies. |
| **Tampering** | In-flight request alteration or SQL injection modifying balances. | Parameterized queries via SQLAlchemy async ORM; strict double-entry invariant validation in Python domain entities (`Σ Debits == Σ Credits`). |
| **Repudiation** | User denies having initiated a transfer or transaction. | Append-only `domain_events` with unique event UUIDs, caller IP metadata, and tamper-resistant timestamps. |
| **Information Disclosure** | Data leakage between tenants or prompt injection exposing unauthorized customer records. | Account-scoped SQL queries; the AI Assistant is restricted to read-only parameterized tools that reject multi-tenant crossovers. |
| **Denial of Service** | Network retry floods or duplicate transfer submission causing overdrafts. | Scoped `Idempotency-Key` headers stored in PostgreSQL transactions. Retries return cached responses without repeating transfers. |
| **Elevation of Privilege** | An attacker prompts the LLM to execute arbitrary SQL or administrative overrides. | The LLM has zero arbitrary SQL execution privileges; it can only invoke strictly typed, read-only Python function tools. |

---

## 3. Financial Integrity Controls

- **Exact Integer Minor Units**: All monetary values are represented as `int` (minor units, e.g. cents) rather than `float` or `double`, eliminating floating-point rounding drift.
- **Atomic Database Boundaries**: Multi-table updates (journal entries + domain events + balance projections) are wrapped in a single database transaction (`BEGIN ... COMMIT`). If any step fails, the entire transaction rolls back cleanly.
- **Idempotency Guard**: Unique database index on `idempotency_records.idempotency_key` ensures network duplicate requests return the original receipt rather than executing a duplicate transfer.
