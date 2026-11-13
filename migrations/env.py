from logging.config import fileConfig

import sqlalchemy as sa
from alembic import context

from app.core.settings import get_settings
from app.core.tables import NODE, PLATFORM, metadata

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

SCHEMAS = (PLATFORM, NODE)


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    if type_ == "schema":
        return name in SCHEMAS
    return True


def url() -> str:
    return config.attributes.get("url") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=url(),
        target_metadata=metadata,
        literal_binds=True,
        include_schemas=True,
        include_name=include_name,
        version_table_schema=PLATFORM,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = sa.create_engine(url(), poolclass=sa.pool.NullPool)
    with engine.connect() as connection:
        for schema in SCHEMAS:
            connection.execute(sa.schema.CreateSchema(schema, if_not_exists=True))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=metadata,
            include_schemas=True,
            include_name=include_name,
            version_table_schema=PLATFORM,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
