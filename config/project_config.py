import os
import yaml
from loguru import logger


class ProjectConfig:
    """
    Loads project configuration from YAML.
    Keeps backward-compatible attribute names.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        logger.info(f"Loading project configuration from: {config_path}")

        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        logger.success("Configuration file loaded")

        self.source_type = self.cfg["source"]["type"]

        self.csv_path = self.cfg["source"].get("csv", {}).get("path")
        self.sql_query = self.cfg["source"].get("sql", {}).get("query")

        self.audio_output_path = self.cfg["audio"]["output_dir"]
        self.audio_url_regex = self.cfg["audio"].get("url_regex", [])
        self.output_path = self.cfg["output"]["output_dir"]

        thresholds = self.cfg.get("thresholds", {})

        self.threshold_demora_inicio = thresholds.get(
            "interaction_start_delay_sec", 5
        )

        self.threshold_silencio_user = thresholds.get(
            "user_silence_sec", 5
        )

        self.threshold_demora_geracao_audio = thresholds.get(
            "audio_generation_delay_sec", 5
        )

        self.openai_api_key_env = self.cfg["env"]["openai_api_key_env"]

        logger.info(
            "Config loaded | "
            f"source_type={self.source_type} | "
            f"audio_output_path={self.audio_output_path}"
        )

    def get_openai_api_key(self) -> str:
        api_key = os.getenv(self.openai_api_key_env)

        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{self.openai_api_key_env}' is not set"
            )

        return api_key

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __repr__(self):
        return f"ProjectConfig(source_type={self.source_type})"
