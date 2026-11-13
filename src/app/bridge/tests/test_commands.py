import sqlalchemy as sa

from app.bridge import commands
from app.core.tables import order_commands


def test_a_command_is_durable_with_its_client_order_id_before_anything_is_sent(
    database: sa.engine.Engine,
) -> None:
    with database.begin() as connection:
        found = commands.submit(
            connection, "submit", {"symbol": "SPUS", "side": "buy", "qty": 1}, "manual"
        )
    assert found.status == "pending" and found.attempts == 0
    assert found.client_order_id == f"CMD-{found.id.hex}"
    with database.begin() as connection:
        assert commands.get(connection, found.id) == found
        assert [event.status for event in commands.events(connection, found.id)] == ["pending"]
        assert commands.waiting(connection) == 1


def test_a_cancel_has_no_client_order_id_of_its_own(database: sa.engine.Engine) -> None:
    with database.begin() as connection:
        found = commands.submit(connection, "cancel", {"client_order_id": "CMD-x"})
    assert found.client_order_id is None


def test_unknown_kinds_are_refused_by_the_database(database: sa.engine.Engine) -> None:
    import uuid

    import pytest

    with pytest.raises(sa.exc.IntegrityError), database.begin() as connection:
        connection.execute(
            sa.insert(order_commands).values(id=uuid.uuid4(), kind="teleport", payload={})
        )


def test_the_halt_flag_is_one_row(database: sa.engine.Engine) -> None:
    with database.begin() as connection:
        assert commands.halted(connection) == (False, None)
        commands.set_halted(connection, True, "manual kill switch")
        commands.set_halted(connection, True, "again")
        assert commands.halted(connection) == (True, "again")
        commands.set_halted(connection, False)
        assert commands.halted(connection) == (False, None)
