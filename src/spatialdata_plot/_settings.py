from dataclasses import dataclass

@dataclass
class Settings:
    verbose: bool = False

settings = Settings()
