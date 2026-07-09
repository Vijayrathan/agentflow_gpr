"""Shared LangGraph Postgres checkpointer for the WebSocket chat pipeline.

One process-wide sync PostgresSaver (the agent's `.invoke` already runs inside
`asyncio.to_thread`, so the sync saver is the right variant). It persists the
LLM message history per `thread_id`, which is what lets a chat resume its
conversation after a backend restart. The CLI and the key-free tests never
touch this module — `SingleAgentSession` falls back to `InMemorySaver` when no
checkpointer factory is passed.

`saver.setup()` creates/updates LangGraph's own tables (checkpoints,
checkpoint_blobs, checkpoint_writes + its migration table) — deliberately NOT
managed by Alembic.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def get_checkpointer():
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    from db.db import pg_dsn

    pool = ConnectionPool(
        pg_dsn(),
        max_size=10,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver
