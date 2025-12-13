# Curation Pipeline

## Overview

This project implements a **modular curation pipeline** designed to analyze conversation data, detect specific behaviors in user-bot interactions, and generate structured outputs for analysis. The pipeline is fully **extensible**, allowing new modules to be added for detecting different behaviors.

The first implemented system is **Delay Interaction**, which focuses on identifying delays and silences in conversations.

---

## Pipeline Architecture

curation/
├── ingestion/              # Download and handle audio files
├── preprocessing/          # Text and audio preprocessing modules
├── curation/               # Behavior detection systems (classes)
│   └── delay_interaction.py
├── config/                 # Project configuration and YAML loader
└── pipeline.py             # Orchestrator script that runs all modules

---

## Key Concepts

- **Ingestion:** Downloads audio files from URLs, saves them locally, and handles temporary storage.
- **Preprocessing:** Cleans text (HTML removal, special characters) and processes audio (channel separation, segment detection).
- **Curation:** Each system is a class implementing logic for detecting a specific behavior. Systems return standardized dataframes.

---

## Delay Interaction System

The **Delay Interaction** system analyzes conversation sessions to detect:

1. **Delayed Interaction Start:** When the first bot response occurs after a configurable threshold.
2. **Long Silences:** When the user takes longer than the configured threshold to respond.
3. **Delayed Audio Generation:** When the generated audio response takes longer than expected compared to message timestamps.

### How It Works

- Loads audio and text for a session.
- Splits stereo audio channels into user and bot channels.
- Detects speech segments and applies silence/delay rules.
- Matches audio segments with text messages using embeddings and semantic similarity.
- Returns a standardized dataframe with the following columns:

| Column              | Description                                                  |
|--------------------|--------------------------------------------------------------|
| timestamp           | Timestamp of the bot message                                  |
| mensagem_bot        | Original bot message                                         |
| matched_text        | Transcribed audio segment                                     |
| tempo_envio_s       | Seconds between user message and bot response                |
| start_time_s        | Segment start time in seconds                                 |
| end_time_s          | Segment end time in seconds                                   |
| score               | Semantic similarity score between text and audio             |
| sessao              | Session ID                                                   |

---

### Example Output

| timestamp           | mensagem_bot                  | matched_text              | tempo_envio_s | start_time_s | end_time_s | score | sessao |
|--------------------|------------------------------|--------------------------|---------------|--------------|------------|-------|--------|
| 2025-12-02 10:00:01 | Hello!                        | Hello!                   | 2.0           | 2.0          | 3.5        | 0.95  | 12345  |
| 2025-12-02 10:00:05 | How can I help you today?     | How can I help you today?| 1.5           | 1.5          | 3.0        | 0.93  | 12345  |

---

## Extensibility

- New behavior detection systems can be added under the `curation/` folder as separate classes.
- The orchestrator (`pipeline.py`) will call each system in sequence, accumulating results.
- Configuration (thresholds, regex patterns, audio paths, OpenAI token, etc.) is managed via `config/config.yaml`.

---

## Running the Pipeline

1. Configure `config/config.yaml` with your CSV/SQL source, audio paths, and OpenAI token.
2. Ensure all required packages are installed (`pip install -r requirements.txt`).
3. Run the pipeline:

```bash
python pipeline.py
