from __future__ import annotations

import hashlib
import secrets
import sqlite3

from app.core.config import AUTH_DB_PATH


ALLOWED_AUTH_ROLES = (
    "Customer",
    "Branch Manager",
    "Risk & Compliance Officer",
    "Admin",
    "Customer Support Agent",
)

EXPECTED_ROLE_SQL = "'Customer', 'Branch Manager', 'Risk & Compliance Officer', 'Admin', 'Customer Support Agent'"


def get_connection(db_path=AUTH_DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    active_salt = salt or secrets.token_hex(16)
    digest = hashlib.sha256(f"{active_salt}:{password}".encode("utf-8")).hexdigest()
    return active_salt, digest


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    _, calculated_hash = hash_password(password, salt=salt)
    return secrets.compare_digest(calculated_hash, password_hash)


def schema_is_current(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
    ).fetchone()
    if row is None or not row["sql"]:
        return False

    schema_sql = row["sql"]
    return (
        "support_agent_id" in schema_sql
        and "branch_manager_id" in schema_sql
        and "relationship_manager_id" not in schema_sql
        and EXPECTED_ROLE_SQL in schema_sql
    )


def rebuild_users_table(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS users")
    connection.commit()


def create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Customer', 'Branch Manager', 'Risk & Compliance Officer', 'Admin', 'Customer Support Agent')),
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            support_agent_id INTEGER REFERENCES users(id),
            branch_manager_id INTEGER REFERENCES users(id),
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()


def upsert_user(
    connection: sqlite3.Connection,
    *,
    username: str,
    full_name: str,
    role: str,
    password: str,
    support_agent_id: int | None = None,
    branch_manager_id: int | None = None,
) -> int:
    salt, password_hash = hash_password(password)
    connection.execute(
        """
        INSERT INTO users (
            username,
            full_name,
            role,
            password_salt,
            password_hash,
            support_agent_id,
            branch_manager_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            full_name = excluded.full_name,
            role = excluded.role,
            password_salt = excluded.password_salt,
            password_hash = excluded.password_hash,
            support_agent_id = excluded.support_agent_id,
            branch_manager_id = excluded.branch_manager_id,
            is_active = 1
        """,
        (
            username,
            full_name,
            role,
            salt,
            password_hash,
            support_agent_id,
            branch_manager_id,
        ),
    )
    connection.commit()
    row = connection.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    return int(row["id"])


def seed_sample_users(connection: sqlite3.Connection) -> None:
    admin_id = upsert_user(
        connection,
        username="admin.anita",
        full_name="Anita Rao",
        role="Admin",
        password="admin123",
    )
    branch_manager_id = upsert_user(
        connection,
        username="branch.raj",
        full_name="Raj Verma",
        role="Branch Manager",
        password="branch123",
    )
    risk_officer_id = upsert_user(
        connection,
        username="risk.neha",
        full_name="Neha Kapoor",
        role="Risk & Compliance Officer",
        password="risk123",
        branch_manager_id=branch_manager_id,
    )
    support_agent_id = upsert_user(
        connection,
        username="support.kiran",
        full_name="Kiran Das",
        role="Customer Support Agent",
        password="support123",
        branch_manager_id=branch_manager_id,
    )
    upsert_user(
        connection,
        username="customer.asha",
        full_name="Asha Menon",
        role="Customer",
        password="customer123",
        support_agent_id=support_agent_id,
        branch_manager_id=branch_manager_id,
    )
    upsert_user(
        connection,
        username="customer.rahul",
        full_name="Rahul Singh",
        role="Customer",
        password="customer456",
        support_agent_id=support_agent_id,
        branch_manager_id=branch_manager_id,
    )
    connection.execute(
        "UPDATE users SET branch_manager_id = ? WHERE id IN (?, ?, ?)",
        (branch_manager_id, admin_id, risk_officer_id, support_agent_id),
    )
    connection.commit()


def initialize_database(seed: bool = True, db_path=AUTH_DB_PATH):
    connection = get_connection(db_path)
    try:
        if not schema_is_current(connection):
            rebuild_users_table(connection)
        create_tables(connection)
        if seed:
            seed_sample_users(connection)
    finally:
        connection.close()
    return db_path


def get_user_by_username(username: str, db_path=AUTH_DB_PATH) -> sqlite3.Row | None:
    connection = get_connection(db_path)
    try:
        return connection.execute(
            """
            SELECT
                u.id,
                u.username,
                u.full_name,
                u.role,
                u.password_salt,
                u.password_hash,
                u.is_active,
                sa.id AS support_agent_id,
                sa.full_name AS support_agent_name,
                sa.username AS support_agent_username,
                bm.id AS branch_manager_id,
                bm.full_name AS branch_manager_name,
                bm.username AS branch_manager_username
            FROM users u
            LEFT JOIN users sa ON sa.id = u.support_agent_id
            LEFT JOIN users bm ON bm.id = u.branch_manager_id
            WHERE u.username = ?
            """,
            (username,),
        ).fetchone()
    finally:
        connection.close()


def authenticate_user(username: str, password: str, db_path=AUTH_DB_PATH) -> dict[str, str] | None:
    row = get_user_by_username(username, db_path=db_path)
    if row is None or not int(row["is_active"]):
        return None
    if not verify_password(password, row["password_salt"], row["password_hash"]):
        return None

    return {
        "id": str(row["id"]),
        "username": row["username"],
        "full_name": row["full_name"],
        "role": row["role"],
        "support_agent_id": str(row["support_agent_id"] or ""),
        "support_agent_name": row["support_agent_name"] or "",
        "support_agent_username": row["support_agent_username"] or "",
        "branch_manager_id": str(row["branch_manager_id"] or ""),
        "branch_manager_name": row["branch_manager_name"] or "",
        "branch_manager_username": row["branch_manager_username"] or "",
    }
