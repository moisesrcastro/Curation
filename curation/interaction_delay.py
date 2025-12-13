import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from preprocessing.text_processing import TextProcessing


class DemoraInteracao:
    """
    Curation system to detect interaction delay patterns:
    - Delay to start speaking
    - Delay in audio generation
    - Long silence between user and bot
    """

    def __init__(
        self,
        audio_base_path: str,
        audio_filename_regex: str,
        bot_values: list[str],
        silence_threshold_seconds: float = 5.0,
        semantic_similarity_threshold: float = 0.4,
    ):
        self.audio_base_path = audio_base_path
        self.audio_filename_regex = audio_filename_regex
        self.bot_values = bot_values

        self.silence_threshold_seconds = silence_threshold_seconds
        self.semantic_similarity_threshold = semantic_similarity_threshold

        self.text_processor = TextProcessing()

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.openai_client = OpenAI()


    def analyze_session(
        self,
        session_id: str,
        df_audio: pd.DataFrame,
        df_messages: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """
        Main entry point for analyzing one session.
        """

        session_audio = df_audio[df_audio["sessao"] == session_id]
        if session_audio.empty:
            return None

        audio_match = session_audio["response"].str.extract(
            self.audio_filename_regex
        )[0]

        if audio_match.empty:
            return None

        audio_path = os.path.join(self.audio_base_path, audio_match.iloc[0])
        if not os.path.exists(audio_path):
            return None

        df_bot = self._load_bot_messages(session_id, df_messages)
        if df_bot.empty:
            return None

        try:
            y_bot, sr = self._load_bot_audio(audio_path)
        except Exception:
            return self._fallback_dataframe(df_bot, session_id)

        audio_segments = self._detect_speech_segments(y_bot, sr)
        segment_transcriptions = self._transcribe_segments(
            audio_segments, y_bot, sr
        )

        return self._semantic_temporal_matching(
            session_id,
            df_bot,
            segment_transcriptions,
        )


    def _load_bot_messages(self, session_id, df_messages):
        df_bot = df_messages[
            (df_messages["sessao"] == session_id)
            & (df_messages["who"].isin(self.bot_values))
        ].copy()

        if df_bot.empty:
            return df_bot

        df_bot = (
            df_bot.drop_duplicates(subset=["timestamp", "text_preenchido"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        df_bot["timestamp"] = pd.to_datetime(df_bot["timestamp"])
        df_bot["clean_text"] = self.text_processor.clean_batch(
            df_bot["text_preenchido"].tolist()
        )

        return df_bot

    def _load_bot_audio(self, audio_path):
        y_stereo, sr = sf.read(audio_path)
        if y_stereo.ndim == 2:
            return y_stereo[:, 1], sr
        return y_stereo, sr


    def _detect_speech_segments(
        self,
        y,
        sr,
        min_duration=0.8,
        merge_gap=0.3,
    ):
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        spectral = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=256
        )[0]

        combined = (
            (rms - rms.min()) / (rms.max() - rms.min() + 1e-9) * 0.7
            + (spectral - spectral.min())
            / (spectral.max() - spectral.min() + 1e-9)
            * 0.3
        )

        threshold = np.percentile(combined, 70) * 0.08
        active = combined > threshold

        changes = np.diff(active.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if active[0]:
            starts = np.insert(starts, 0, 0)
        if active[-1]:
            ends = np.append(ends, len(active))

        segments = []
        for s, e in zip(starts, ends):
            start_t = s * 256 / sr
            end_t = e * 256 / sr
            if end_t - start_t >= min_duration:
                segments.append(
                    {"start": start_t, "end": end_t}
                )

        return segments

    def _transcribe_segments(self, segments, y_audio, sr):
        results = []

        for idx, seg in enumerate(segments):
            start = int(seg["start"] * sr)
            end = int(seg["end"] * sr)

            audio_slice = y_audio[start:end]
            if len(audio_slice) == 0:
                continue

            tmp_path = f"tmp_segment_{idx}.wav"
            sf.write(tmp_path, audio_slice, sr)

            try:
                with open(tmp_path, "rb") as f:
                    text = self.openai_client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f,
                        response_format="text",
                    )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            results.append(
                {
                    "segment_index": idx,
                    "start_time": seg["start"],
                    "end_time": seg["end"],
                    "transcription": text.strip(),
                }
            )

        return results


    def _semantic_temporal_matching(
        self,
        session_id,
        df_bot,
        segment_transcriptions,
    ):
        embeddings = self.embed_model.encode(
            df_bot["clean_text"].tolist(),
            convert_to_tensor=True,
        )

        start_ts = df_bot["timestamp"].min()
        message_times = (
            df_bot["timestamp"] - start_ts
        ).dt.total_seconds()

        rows = []
        used_messages = set()

        for seg in segment_transcriptions:
            if len(seg["transcription"]) < 5:
                continue

            seg_emb = self.embed_model.encode(
                [seg["transcription"]],
                convert_to_tensor=True,
            )

            sims = util.cos_sim(embeddings, seg_emb)[:, 0].cpu().numpy()

            best_idx = -1
            best_score = self.semantic_similarity_threshold

            for i, score in enumerate(sims):
                if i in used_messages:
                    continue

                time_diff = seg["start_time"] - message_times.iloc[i]
                if time_diff >= -2 and score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx != -1:
                used_messages.add(best_idx)

                rows.append(
                    {
                        "session": session_id,
                        "timestamp": df_bot.iloc[best_idx]["timestamp"],
                        "bot_message": df_bot.iloc[best_idx]["clean_text"],
                        "matched_audio": seg["transcription"],
                        "audio_start_s": round(seg["start_time"], 3),
                        "similarity": round(best_score, 3),
                    }
                )

        return pd.DataFrame(rows)


    def _fallback_dataframe(self, df_bot, session_id):
        rows = []
        start_ts = df_bot["timestamp"].min()

        for _, row in df_bot.iterrows():
            rows.append(
                {
                    "session": session_id,
                    "timestamp": row["timestamp"],
                    "bot_message": row["clean_text"],
                    "matched_audio": None,
                    "audio_start_s": None,
                    "similarity": None,
                }
            )

        return pd.DataFrame(rows)
