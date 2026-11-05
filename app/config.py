from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="VA_", extra="ignore")

    database_url: str = "sqlite:///./voice_agent.db"
    webhook_timeout_seconds: float = 5.0
    max_dial_attempts: int = 3
    retry_delay_minutes: int = 30
    default_timezone: str = "Asia/Kolkata"
    default_max_concurrency: int = 40
    # Public URL where Plivo can reach this server (ngrok or production domain).
    public_url: str = "http://localhost:8000"
    caller_id: str = ""
    stream_secret: str = "dev-stream-secret-change-me"


@lru_cache
def get_settings() -> Settings:
    return Settings()
