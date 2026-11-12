<#
.SYNOPSIS
    LedgerMind AI - Operational CLI Launcher and Orchestrator

.DESCRIPTION
    Provides unified command-line orchestration for starting services,
    running test suites, seeding demonstration accounts, and auditing
    accounting invariant integrity.

.EXAMPLE
    .\scripts\run_app.ps1 api
    .\scripts\run_app.ps1 ui
    .\scripts\run_app.ps1 test
    .\scripts\run_app.ps1 lint
    .\scripts\run_app.ps1 seed
    .\scripts\run_app.ps1 audit
#>

param (
    [Parameter(Position = 0, Mandatory = $false)]
    [ValidateSet("api", "ui", "all", "test", "lint", "seed", "audit", "help")]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location -Path $ProjectRoot

# Locate Python binary (virtual environment preferred)
$PythonBin = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonBin)) {
    $PythonBin = "python"
}

function Show-Header {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "   LedgerMind AI - Financial Analytics & Fraud Platform   " -ForegroundColor Yellow
    Write-Host "==========================================================" -ForegroundColor Cyan
}

switch ($Command) {
    "api" {
        Show-Header
        Write-Host "Starting FastAPI Backend Server on http://localhost:8000..." -ForegroundColor Green
        & $PythonBin -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "ui" {
        Show-Header
        Write-Host "Starting Streamlit Operations Cockpit on http://localhost:8501..." -ForegroundColor Green
        & $PythonBin -m streamlit run app/dashboard.py --server.port 8501
    }
    "test" {
        Show-Header
        Write-Host "Running comprehensive test suite (Unit, Integration, API, UI)..." -ForegroundColor Green
        & $PythonBin -m pytest tests/unit/ tests/integration/ tests/api/ -v
    }
    "lint" {
        Show-Header
        Write-Host "Executing Ruff static analysis & code style audits..." -ForegroundColor Green
        & $PythonBin -m ruff check app/ ml/ tests/ scripts/
    }
    "seed" {
        Show-Header
        Write-Host "Seeding institutional accounts, transfers, and demo balances..." -ForegroundColor Green
        & $PythonBin scripts/seed_demo_data.py
    }
    "audit" {
        Show-Header
        Write-Host "Auditing Event-Sourced Double-Entry Invariant (Sigma Debits == Sigma Credits)..." -ForegroundColor Green
        $AuditScript = @"
import sqlite3
import os

db_path = 'ledgermind.db'
if not os.path.exists(db_path):
    print('[WARN] Database file ledgermind.db not found.')
    exit(0)

conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute('SELECT entry_type, SUM(amount_minor_units) FROM journal_entries GROUP BY entry_type')
totals = dict(c.fetchall())
debits = totals.get('DEBIT', 0)
credits = totals.get('CREDIT', 0)
drift = debits - credits

print(f'Total Journal Debits : {debits / 100:,.2f}')
print(f'Total Journal Credits: {credits / 100:,.2f}')
print(f'Accounting Drift     : {drift / 100:,.2f}')

if drift == 0 and debits > 0:
    print('[SUCCESS] Double-Entry Accounting Invariant verified: PERFECT EQUILIBRIUM (drift == 0).')
elif drift == 0:
    print('[INFO] Zero entries recorded yet.')
else:
    print(f'[FAIL] Invariant violated! Imbalance: {drift}')
    exit(1)
"@
        & $PythonBin -c $AuditScript
    }
    Default {
        Show-Header
        Write-Host "Usage: .\scripts\run_app.ps1 <command>" -ForegroundColor White
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Yellow
        Write-Host "  api    - Launch FastAPI REST API server (port 8000)"
        Write-Host "  ui     - Launch Streamlit Operations Cockpit (port 8501)"
        Write-Host "  test   - Run full Pytest suite"
        Write-Host "  lint   - Run Ruff linter and static code analysis"
        Write-Host "  seed   - Seed database with realistic institutional accounts and activity"
        Write-Host "  audit  - Verify double-entry ledger equilibrium invariant (Debits == Credits)"
        Write-Host "  help   - Show this reference menu"
        Write-Host ""
    }
}
