"""FastAPI app entrypoint for TalkingHeadBench OpenEnv server."""

from __future__ import annotations

import sys
from pathlib import Path

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.talking_head_environment import TalkingHeadEnvironment

app = create_app(
    TalkingHeadEnvironment,
    Action,
    Observation,
    env_name="talking_head_bench",
)


def main() -> None:
    """Run the TalkingHeadBench environment server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()