from __future__ import annotations

from app.auth.database import AUTH_DB_PATH, initialize_database


def main() -> None:
    db_path = initialize_database(seed=True)
    print(f"Authentication database ready: {db_path}")
    print("Sample users:")
    print("- admin.anita / admin123")
    print("- branch.raj / branch123")
    print("- risk.neha / risk123")
    print("- support.kiran / support123")
    print("- customer.asha / customer123")
    print("- customer.rahul / customer456")
