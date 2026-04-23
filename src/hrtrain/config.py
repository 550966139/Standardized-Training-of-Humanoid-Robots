"""Runtime configuration."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HRTRAIN_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 6006
    reload: bool = False

    data_root: Path = Field(default=Path("data"))
    db_url: str = "sqlite+aiosqlite:///data/hrtrain.db"

    # Upstream dependencies (paths on the AutoDL host)
    isaaclab_root: Path = Field(default=Path("/root/autodl-tmp/IsaacLab"))
    unitree_rl_lab_root: Path = Field(default=Path("/root/autodl-tmp/unitree_rl_lab"))
    hc_root: Path = Field(default=Path("/root/autodl-tmp/humanoid-choreo"))
    conda_env_isaaclab: str = "hc-isaac"
    conda_env_gmr: str = "gmr"
    conda_root: Path = Field(default=Path("/root/miniconda3"))

    blender_bin: str = "blender"

    # File upload limit (bytes) — 500 MB default
    max_upload_bytes: int = 500 * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.data_root / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_root / "jobs"

    @property
    def outputs_dir(self) -> Path:
        return self.data_root / "outputs"

    def ensure_dirs(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
