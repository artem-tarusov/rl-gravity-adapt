import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import uuid4

import yaml

from gravity_adapt.config import Config


@dataclass(frozen=True)
class Run:
    directory: Path
    _started: float = field(default_factory=monotonic, init=False)

    def elapsed_seconds(self) -> float:
        return monotonic() - self._started


def create_run(config: Config) -> Run:
    started_at = datetime.now(UTC)
    suffix = uuid4().hex[:8]
    run_id = f"{started_at:%Y-%m-%d_%H-%M-%SZ}_{config.run.name}_s{config.run.seed}_{suffix}"
    output = Path(config.run.output_dir).expanduser().resolve()
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    run = Run(directory)
    resolved = asdict(config)
    resolved["run"]["output_dir"] = str(output)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": started_at.isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("gymnasium", "mujoco", "torch", "numpy")},
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return run


def finish_run(
    run: Run,
    *,
    status: Literal["completed", "failed", "interrupted"],
    result: dict | None = None,
    error: str | None = None,
) -> None:
    path = run.directory / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(
        status=status,
        finished_at=datetime.now(UTC).isoformat(),
        duration_seconds=run.elapsed_seconds(),
    )
    if result is not None:
        metadata["result"] = result
    if error is not None:
        metadata["error"] = error
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
