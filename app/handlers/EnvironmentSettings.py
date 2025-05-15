from pydantic import BaseSettings

class Settings(BaseSettings):
    secret_key: str
    debug_mode: bool = False

    class Config:
        env_file = ".env"
