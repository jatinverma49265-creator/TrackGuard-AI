"""PostgreSQL-only migration smoke test; SQLite is intentionally unsupported."""

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


@pytest.fixture
def migrated_postgres_schema():
    database_url = os.environ.get("TRACKGUARD_TEST_DATABASE_URL")
    disposable = os.environ.get("TRACKGUARD_TEST_DATABASE_DISPOSABLE")
    if not database_url or disposable != "YES":
        pytest.skip(
            "Set TRACKGUARD_TEST_DATABASE_URL and TRACKGUARD_TEST_DATABASE_DISPOSABLE=YES "
            "for an explicitly disposable PostgreSQL database"
        )
    if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("TRACKGUARD_TEST_DATABASE_URL must point to PostgreSQL")

    schema_name = f"trackguard_test_{uuid4().hex}"
    admin_engine = create_engine(database_url, pool_pre_ping=True)
    schema_created = False
    try:
        with admin_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
        schema_created = True

        schema_engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args={"options": f"-csearch_path={schema_name}"},
        )
        config = Config("alembic.ini")
        with schema_engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield schema_engine
    finally:
        if "schema_engine" in locals():
            schema_engine.dispose()
        if schema_created:
            with admin_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{schema_name}" CASCADE'))
        admin_engine.dispose()


def test_initial_migration_creates_tenant_scoped_tables(migrated_postgres_schema):
    table_names = set(inspect(migrated_postgres_schema).get_table_names())
    assert {
        "users",
        "tenants",
        "videos",
        "processing_jobs",
        "event_candidates",
        "evidence_assets",
        "event_reviews",
        "processing_logs",
        "alembic_version",
    } <= table_names
