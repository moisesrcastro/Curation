import os
import requests
import pandas as pd
from loguru import logger
from typing import List, Optional


class AudioIngestion:
    """
    Handles audio URL extraction, download and cleanup.
    """

    def __init__(
        self,
        config,
        response_col: str = "response",
        timeout: int = 30
    ):
        self.config = config
        self.response_col = response_col
        self.timeout = timeout

        self.output_path = config.audio_output_path
        self.url_regex_list = config.audio_url_regex

        if not self.url_regex_list:
            raise ValueError("Audio URL regex list is empty")

        os.makedirs(self.output_path, exist_ok=True)
        logger.info("AudioIngestion initialized")

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract audio URLs from dataframe.
        """
        logger.info("Extracting audio URLs")

        if self.response_col not in df.columns:
            raise ValueError(f"Column '{self.response_col}' not found")

        df_audio = df.copy()
        df_audio["audio_url"] = None

        for regex in self.url_regex_list:
            extracted = df_audio[self.response_col].str.extract(regex)
            df_audio["audio_url"] = df_audio["audio_url"].fillna(extracted[0])

        df_audio = df_audio[df_audio["audio_url"].notna()]
        logger.info(f"Extracted {df_audio.shape[0]} audio URLs")

        return df_audio

    def download(
        self,
        df_audio: pd.DataFrame,
        url_col: str = "audio_url"
    ) -> List[str]:
        """
        Download all audio files from dataframe.
        """
        logger.info("Downloading audio files")

        downloaded_files = []

        for _, row in df_audio.iterrows():
            url = row[url_col]
            filename = url.split("/")[-1]
            file_path = os.path.join(self.output_path, filename)

            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(response.content)

                downloaded_files.append(file_path)
                logger.debug(f"Downloaded audio: {file_path}")

            except Exception as e:
                logger.error(f"Failed to download audio {url}: {e}")

        logger.info(f"Downloaded {len(downloaded_files)} audio files")
        return downloaded_files

    def cleanup(self, file_paths: List[str]):
        """
        Remove downloaded audio files.
        """
        logger.info("Cleaning up audio files")

        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove file {path}: {e}")
