"""Environment configuration for the NBA Player Prediction API."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Settings:
    data_dir: Path
    allowed_origins: List[str]
    env: str
    reload: bool
    static_dir: Optional[Path] = None

    @property
    def frontend_home_path(self) -> Path:
        return self.data_dir / "frontend_home.parquet"

    @property
    def frontend_all_path(self) -> Path:
        return self.data_dir / "frontend_all.parquet"

    @property
    def pipeline_status_path(self) -> Path:
        return self.data_dir / "pipeline_status.json"

    @property
    def pipeline_runs_path(self) -> Path:
        return self.data_dir / "pipeline_runs.jsonl"


def get_settings() -> Settings:
    raw_data_dir = os.environ.get("PROPREDICT_DATA_DIR", "./data")
    raw_origins = os.environ.get("PROPREDICT_ALLOWED_ORIGINS", "http://localhost:5173")
    origins = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]
    if not origins:
        origins = ["http://localhost:5173"]

    env = os.environ.get("PROPREDICT_ENV", "development")
    reload_str = os.environ.get("PROPREDICT_RELOAD", "false").lower()
    reload = reload_str in ("true", "1", "yes")

    raw_static_dir = os.environ.get("PROPREDICT_STATIC_DIR", "./lineup-builder/dist")
    static_dir_path = Path(raw_static_dir)
    static_dir = static_dir_path if static_dir_path.is_dir() else None

    return Settings(
        data_dir=Path(raw_data_dir),
        allowed_origins=origins,
        env=env,
        reload=reload,
        static_dir=static_dir,
    )
