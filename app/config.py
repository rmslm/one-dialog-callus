from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    openai_api_key: str = os.getenv('OPENAI_API_KEY')

    class Config:
        env_file = ".env"


settings = Settings()