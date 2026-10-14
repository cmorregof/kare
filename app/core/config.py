"""Configuración leída del entorno o de un archivo .env.

Ninguna clave se imprime ni se registra. Sin claves, el sistema funciona en
modo solo recuperación: muestra los pasajes del corpus y no genera texto.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai_compat", "none"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Modelo de lenguaje --------------------------------------------------
    # "anthropic": API de Claude. "openai_compat": OpenAI, OpenRouter, Groq u
    # Ollama local (http://localhost:11434/v1), con el que la pregunta no sale
    # del equipo. "none": solo recuperación.
    llm_provider: LLMProvider = "anthropic"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-opus-5-5"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = ""
    # Algunos servidores compatibles no aceptan response_format y algunos
    # modelos de OpenAI piden max_completion_tokens en lugar de max_tokens.
    openai_json_mode: bool = True
    openai_max_tokens_field: str = "max_tokens"
    # Modelo del verificador. Vacío = el mismo del generador.
    verifier_model: str = ""
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 90.0
    # None = valor por defecto del proveedor.
    llm_temperature: float | None = None

    # --- Transcripción (solo preguntas dictadas, nunca audio con pacientes) ---
    elevenlabs_api_key: str = ""
    elevenlabs_base_url: str = "https://api.elevenlabs.io"
    elevenlabs_stt_model: str = "scribe_v2"
    # True envía enable_logging=false: modo de retención cero, sin guardar audio
    # ni transcripción. Según la API, solo para clientes Enterprise.
    elevenlabs_zero_retention: bool = False
    transcription_language: str = "es"
    keyterms_path: str = "config/keyterms_paliativos.txt"
    # Respaldo con Whisper de OpenAI; usa openai_api_key.
    whisper_base_url: str = "https://api.openai.com/v1"
    whisper_model: str = "whisper-1"
    transcription_timeout_seconds: float = 120.0

    # --- Corpus ---------------------------------------------------------------
    manifest_path: str = "corpus/manifest.yaml"
    corpus_dir: str = "corpus/fuentes"
    db_path: str = "data/kare.db"
    chunk_target_chars: int = 1000
    chunk_max_chars: int = 1400
    chunk_overlap_chars: int = 150

    # --- Recuperación y abstención ----------------------------------------------
    top_k: int = 8
    per_query_k: int = 20
    # Fracción mínima de los términos de una consulta que debe aparecer en un
    # pasaje para considerarlo pertinente. Si ningún pasaje la alcanza, el
    # sistema se abstiene sin llamar al generador.
    min_coverage: float = 0.3
    # False: solo se muestran afirmaciones que el verificador juzga plenamente
    # sustentadas. True: también las parcialmente sustentadas, marcadas.
    keep_partial_claims: bool = False

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
