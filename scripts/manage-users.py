#!/usr/bin/env python3
"""Create manually provisioned web users without exposing passwords to the browser."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.auth import configured_user_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Local AI Agent web account")
    parser.add_argument("role", choices=("admin", "guest"))
    parser.add_argument("username")
    args = parser.parse_args()
    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("passwords do not match")
    user = configured_user_store().create(args.username, password, args.role)
    print(f"created {user.role} account: {user.username}")


if __name__ == "__main__":
    main()