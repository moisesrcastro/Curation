import pandas as pd
from tqdm import tqdm
from loguru import logger

from config.project_config import ProjectConfig
from ingestion.audio_ingestion import AudioIngestion
from preprocessing.text_processing import TextProcessing
from curation.interaction_delay import InteractionDelayAnalyzer


def load_data(config: ProjectConfig) -> pd.DataFrame:
    logger.info("Loading input data")

    if config.source.type == "csv":
        return pd.read_csv(config.source.path)

    if config.source.type == "sql":
        return config.sql_loader()

    raise ValueError(f"Unsupported source type: {config.source.type}")


def main() -> pd.DataFrame:
    logger.info("Starting Interaction Delay Pipeline")

    config = ProjectConfig.from_yaml("config/config.yaml")

    df = load_data(config)

    audio_ingestion = AudioIngestion(config)
    text_processing = TextProcessing(config)
    analyzer = InteractionDelayAnalyzer(config, text_processing)

    session_col = config.columns.session_id
    audio_url_col = config.columns.audio_url

    results = []

    for session_id in tqdm(
        df[session_col].dropna().unique(),
        desc="Processing sessions"
    ):
        try:
            df_session = df[df[session_col] == session_id]

            audio_path = audio_ingestion.download_audio_for_session(
                df_session,
                audio_url_col
            )

            if audio_path is None:
                continue

            df_result = analyzer.analyze(
                session_id=session_id,
                df_audio=df_session,
                df_messages=df
            )

            if df_result is not None and not df_result.empty:
                results.append(df_result)

            audio_ingestion.cleanup(audio_path)

        except Exception as e:
            logger.error(f"Session {session_id} failed: {e}", exc_info=True)

    if not results:
        logger.warning("No results generated")
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)

    logger.info(f"Pipeline finished. Rows generated: {len(final_df)}")

    return final_df


if __name__ == "__main__":
    output_df = main()
