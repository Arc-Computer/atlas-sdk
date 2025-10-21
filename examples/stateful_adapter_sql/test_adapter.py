import pytest

from atlas.connectors.python import PythonAdapter
from atlas.config.models import AdapterType, PythonAdapterConfig
from atlas.connectors.registry import SessionContext


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_sqlite_agent_session():
    config = PythonAdapterConfig(
        type=AdapterType.PYTHON,
        name="sqlite-agent",
        system_prompt="",
        import_path="examples.stateful_adapter_sql.adapter",
        attribute="SQLiteAgent",
    )
    adapter = PythonAdapter(config)
    session = await adapter.open_session(SessionContext(task_id="sqlite", execution_mode="stepwise"))
    rows = await session.step("SELECT title FROM documents ORDER BY id")
    assert rows["content"][0][0] == "Atlas SDK"
    metadata = await session.close(reason="test")
    assert metadata["closed"] is True
