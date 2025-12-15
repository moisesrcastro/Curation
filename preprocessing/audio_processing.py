import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class AudioProcessor:
    def __init__(
        self,
        min_duration_segment=0.8,
        merge_gap=0.3,
        limiar_silencio_segundos=5,
        top_db_user=25,
        min_duracao_user=0.3,
    ):
        self.min_duration_segment = min_duration_segment
        self.merge_gap = merge_gap
        self.limiar_silencio_segundos = limiar_silencio_segundos
        self.top_db_user = top_db_user
        self.min_duracao_user = min_duracao_user

    def load_audio(self, caminho_audio):
        if not os.path.exists(caminho_audio):
            raise FileNotFoundError(caminho_audio)

        y, sr = sf.read(caminho_audio)
        return y, sr

    def detectar_falas(self, y, sr, top_db=25, min_duracao=0.0):
        non_silent_intervals = librosa.effects.split(y, top_db=top_db)

        segmentos = []
        for start, end in non_silent_intervals:
            duracao = (end - start) / sr
            if duracao >= min_duracao:
                segmentos.append(
                    {
                        "start": start / sr,
                        "end": end / sr,
                        "duration": duracao,
                    }
                )
        return segmentos

    def detect_speech_segments_optimized(self, y, sr):
        rms = librosa.feature.rms(
            y=y, frame_length=1024, hop_length=256
        )[0]
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=256
        )[0]

        rms_min, rms_max = np.min(rms), np.max(rms)
        spectral_min, spectral_max = (
            np.min(spectral_centroid),
            np.max(spectral_centroid),
        )

        rms_norm = (rms - rms_min) / (rms_max - rms_min + 1e-9)
        spectral_norm = (spectral_centroid - spectral_min) / (
            spectral_max - spectral_min + 1e-9
        )

        combined_signal = rms_norm * 0.7 + spectral_norm * 0.3
        threshold = np.percentile(combined_signal, 70) * 0.08

        above_threshold = combined_signal > threshold
        changes = np.diff(above_threshold.astype(int))

        segment_starts = np.where(changes == 1)[0] + 1
        segment_ends = np.where(changes == -1)[0] + 1

        if above_threshold[0]:
            segment_starts = np.insert(segment_starts, 0, 0)
        if above_threshold[-1]:
            segment_ends = np.append(segment_ends, len(above_threshold))

        segments = []
        for start_idx, end_idx in zip(segment_starts, segment_ends):
            start_time = start_idx * 256 / sr
            end_time = end_idx * 256 / sr
            duration = end_time - start_time

            if duration >= self.min_duration_segment:
                segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "duration": duration,
                    }
                )

        if not segments:
            return []

        merged_segments = [segments[0]]
        for seg in segments[1:]:
            last_seg = merged_segments[-1]
            gap = seg["start"] - last_seg["end"]

            if gap < self.merge_gap:
                last_seg["end"] = seg["end"]
                last_seg["duration"] = (
                    last_seg["end"] - last_seg["start"]
                )
            else:
                merged_segments.append(seg)

        return merged_segments

    def analisa_intervalos_silencio(self, caminho_audio):
        if not os.path.exists(caminho_audio):
            return None

        try:
            y_stereo, sr = sf.read(caminho_audio)
            if y_stereo.ndim != 2:
                return None

            canal_user = y_stereo[:, 0]
            canal_bot = y_stereo[:, 1]
            audio_duration = len(canal_user) / sr

        except Exception:
            return None

        seg_user = self.detectar_falas(
            canal_user,
            sr,
            top_db=self.top_db_user,
            min_duracao=self.min_duracao_user,
        )

        seg_bot = self.detectar_falas(
            canal_bot,
            sr,
            top_db=25,
            min_duracao=0.0,
        )

        silencios_validos = []
        for u in seg_user:
            futuros_bot = [b for b in seg_bot if b["start"] > u["end"]]
            if futuros_bot:
                b = futuros_bot[0]
                duracao_silencio = b["start"] - u["end"]
                if duracao_silencio >= self.limiar_silencio_segundos:
                    silencios_validos.append(
                        {
                            "user_fim": u["end"],
                            "bot_ini": b["start"],
                            "duration": duracao_silencio,
                        }
                    )

        return {
            "duracao_total": audio_duration,
            "segmentos_user": len(seg_user),
            "segmentos_bot": len(seg_bot),
            "silencios_user_bot": len(silencios_validos),
            "detalhes_silencios": silencios_validos,
        }

    def separar_estereo(
        self,
        caminho_audio,
        pasta_saida="Audios Divididos",
        mostrar_grafico=True,
    ):
        os.makedirs(pasta_saida, exist_ok=True)

        audio = AudioSegment.from_file(caminho_audio)

        samples = np.array(audio.get_array_of_samples())
        samples = samples.reshape((-1, audio.channels))

        canal_esq = samples[:, 0]
        canal_dir = samples[:, 1]

        caminho_esq = os.path.join(pasta_saida, "canal_esquerdo.wav")
        caminho_dir = os.path.join(pasta_saida, "canal_direito.wav")

        AudioSegment(
            canal_esq.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=1,
        ).export(caminho_esq, format="wav")

        AudioSegment(
            canal_dir.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=1,
        ).export(caminho_dir, format="wav")

        fig = self.gerar_graficos_audio(
            canal_esq,
            canal_dir,
            audio.frame_rate,
        )

        if mostrar_grafico:
            fig.show()

        return caminho_esq, caminho_dir

    def gerar_graficos_audio(
        self,
        canal_esq,
        canal_dir,
        frame_rate,
    ):
        duracao_total = len(canal_esq) / frame_rate
        tempo = np.linspace(0, duracao_total, len(canal_esq))

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Canal Esquerdo", "Canal Direito"),
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Scatter(x=tempo, y=canal_esq, mode="lines"),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=tempo, y=canal_dir, mode="lines"),
            row=2,
            col=1,
        )

        fig.update_layout(
            title_text=f"Forma de Onda dos Canais (Duração: {duracao_total:.2f}s)",
            height=600,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Tempo (s)")
        fig.update_yaxes(title_text="Amplitude")

        return fig
