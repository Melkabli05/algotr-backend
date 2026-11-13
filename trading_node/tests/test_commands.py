import uuid

import sqlalchemy as sa

from app.bridge import commands as writer
from trading_node import commands


def test_two_nodes_claiming_at_once_never_take_the_same_command(
    database: sa.engine.Engine,
) -> None:
    with database.begin() as connection:
        made = [writer.submit(connection, "submit", {"n": n}) for n in range(5)]
    first = database.connect()
    second = database.connect()
    try:
        one = first.begin()
        taken_first = commands.claim(first, "node-a", limit=3)
        taken_second = commands.claim(second, "node-b", limit=10)
        one.commit()
        second.commit()
    finally:
        first.close()
        second.close()
    ids_first = {row["id"] for row in taken_first}
    ids_second = {row["id"] for row in taken_second}
    assert len(ids_first) == 3 and len(ids_second) == 2
    assert not ids_first & ids_second
    assert ids_first | ids_second == {command.id for command in made}


def test_claiming_counts_attempts_and_a_requeued_command_can_be_claimed_again(
    database: sa.engine.Engine,
) -> None:
    with database.begin() as connection:
        made = writer.submit(connection, "submit", {})
        [row] = commands.claim(connection, "node-a")
        assert row["attempts"] == 1 and row["claimed_by"] == "node-a"
        assert [item["id"] for item in commands.unfinished(connection)] == [made.id]
        commands.requeue(connection, made.id, "node restarted before sending")
        [again] = commands.claim(connection, "node-b")
        assert again["attempts"] == 2
        commands.mark(connection, made.id, "done", result={"status": "filled"})
        assert commands.unfinished(connection) == []
        statuses = [event.status for event in writer.events(connection, made.id)]
    assert statuses == ["pending", "claimed", "pending", "claimed", "done"]


def test_client_order_ids_lead_back_to_their_command() -> None:
    command_id = uuid.uuid4()
    assert commands.command_id_of(f"CMD-{command_id.hex}") == command_id
    assert commands.command_id_of(f"CMD-{command_id.hex}-3") == command_id
    assert commands.command_id_of("O-20261107-001") is None
