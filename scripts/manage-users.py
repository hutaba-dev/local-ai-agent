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
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create", help="create an admin or guest account")
    create_parser.add_argument("role", choices=("admin", "guest"))
    create_parser.add_argument("username")
    password_parser = subparsers.add_parser("set-password", help="replace an existing account password")
    password_parser.add_argument("username")
    args = parser.parse_args()
    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("passwords do not match")
    try:
        store = configured_user_store()
        if args.command == "set-password":
            store.set_password(args.username, password)
            print(f"updated password for: {args.username}")
            return
        user = store.create(args.username, password, args.role)
    except ValueError as error:
        raise SystemExit(f"cannot update user: {error}") from error
    print(f"created {user.role} account: {user.username}")


if __name__ == "__main__":
    main()