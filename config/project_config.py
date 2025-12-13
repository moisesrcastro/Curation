import yaml
import os
import pandas as pd
from loguru import logger


class ProjectConfig:
    """
    Loads project configuration from a YAML file.
    Supports local CSV or SQL as data source.
    """

    def __init__(self, config_path="config/config.yaml"):
        logger.info(f"Initializing ProjectConfig using config file: {config_path}")

        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        logger.success("Configuration file loaded successfully")

        self.source_type = self.cfg.get("source", {}).get("type", "csv")
        self.csv_path = self.cfg.get("source", {}).get("csv_path", "data/dados.csv")
        self.sql_query = self.cfg.get("source", {}).get("sql_query", "")
        self.sql_connection_string = self.cfg.get("source", {}).get("sql_connection_string", "")

        self.openai_token = self.cfg.get("openai_token", "")

        self.audio_path = self.cfg.get("audio_path", "data/audio/")
        self.output_path = self.cfg.get("output_path", "data/output/")

        self.threshold_demora_inicio = self.cfg.get("threshold_demora_inicio", 5)
        self.threshold_silencio_user = self.cfg.get("threshold_silencio_user", 5)
        self.threshold_demora_geracao_audio = self.cfg.get("threshold_demora_geracao_audio", 5)

        logger.info(
            f"Configuration loaded | source_type={self.source_type} | "
            f"audio_path={self.audio_path}"
        )

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __repr__(self):
        return f"ProjectConfig(source_type={self.source_type})"

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data | source_type={self.source_type}")

        if self.source_type == "csv":
            if not os.path.exists(self.csv_path):
                logger.error(f"CSV file not found: {self.csv_path}")
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

            df = pd.read_csv(self.csv_path)
            logger.success(f"CSV loaded successfully | rows={len(df)}")
            return df

        elif self.source_type == "sql":
            if not self.sql_query:
                logger.error("sql_query not provided for SQL source")
                raise ValueError("For SQL source, 'sql_query' must be provided")


            df = pd.read_sql(self.sql_query)
            logger.success(f"SQL query executed successfully | rows={len(df)}")
            return df

        else:
            logger.error(f"Unknown source type: {self.source_type}")
            raise ValueError(f"Unknown source type: {self.source_type}")
