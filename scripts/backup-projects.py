#!/usr/bin/env python3
"""Create a bounded SQLite online backup on the dedicated backup HDD."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


SOURCE = Path(os.getenv("PROJECT_DATABASE_PATH", "/var/lib/local-ai-agent/projects.db"))
BACKUP_ROOT = Path(os.getenv("PROJECT_BACKUP_ROOT", "/srv/local-ai-backup/metadata"))
RETENTION = int(os.getenv("PROJECT_BACKUP_RETENTION", "14"))


def main() -> None:
    mount = BACKUP_ROOT.parent
    if not os.path.ismount(mount):
        raise SystemExit(f"backup storage is offline: {mount}")
    if not SOURCE.is_file():
        raise SystemExit(f"project database does not exist: {SOURCE}")
    BACKUP_ROOT.mkdir(mode=0o750, parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = BACKUP_ROOT / f"projects-{timestamp}.db"
    temporary = destination.with_suffix(".tmp")
    with sqlite3.connect(SOURCE) as source, sqlite3.connect(temporary) as target:
        source.backup(target)
        result = target.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError("backup integrity check failed")
    temporary.chmod(0o600)
    temporary.replace(destination)
    backups = sorted(BACKUP_ROOT.glob("projects-*.db"), reverse=True)
    for expired in backups[max(1, RETENTION):]:
        expired.unlink()
    print(destination)


if __name__ == "__main__":
    main()
