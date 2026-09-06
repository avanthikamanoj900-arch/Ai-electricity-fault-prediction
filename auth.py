"""
auth.py — backend auth logic for ElectroGuard AI.

This is a lightweight, in-memory demo user store. It lives at module scope
so it persists for as long as the Streamlit server process is running (i.e.
shared across sessions on that process) — but it is NOT persistent storage:
restarting the app clears all accounts.

For real use, swap _USERS and the functions below for a proper backend:
- a database (SQLite/Postgres) with hashed passwords (e.g. via `bcrypt`)
- or a package like `streamlit-authenticator`
- or an external auth provider (Firebase Auth, Supabase Auth, Auth0, etc.)
"""

from typing import Tuple

# Demo user store: username -> password (plaintext, demo only — never do
# this in production, always hash passwords).
_USERS = {
    "admin": "admin123",
}


def user_exists(username: str) -> bool:
    """Return True if a username is already registered."""
    return username in _USERS


def verify_user(username: str, password: str) -> bool:
    """Check a username/password pair against the store."""
    if not username or not password:
        return False
    return _USERS.get(username) == password


def register_user(username: str, password: str, confirm_password: str) -> Tuple[bool, str]:
    """
    Attempt to create a new account.

    Returns (success, message).
    """
    username = (username or "").strip()

    if not username or not password:
        return False, "Please fill in both a username and a password."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    if user_exists(username):
        return False, "That username is already taken."

    _USERS[username] = password
    return True, "Account created — you can log in now."


def login(username: str, password: str) -> Tuple[bool, str]:
    """
    Attempt to log in.

    Returns (success, message).
    """
    if verify_user(username, password):
        return True, f"Welcome back, {username}."
    return False, "Incorrect username or password."
