import numpy as np
import librosa
from typing import Tuple, Optional
from pathlib import Path
import logging


class AudioPreprocessor:
    """Tiền xử lý và trích xuất đặc trưng file âm thanh cho bài toán tìm kiếm"""

    def __init__(self, sr: int = 22050, n_mfcc: int = 20, hop_length: int = 512):
        """
        Args:
            sr         : Sample rate
            n_mfcc     : Số hệ số MFCC
            hop_length : Bước nhảy frame (samples)
        """
        self.sr         = sr
        self.n_mfcc     = n_mfcc
        self.hop_length = hop_length

    # ── 1. Load ──────────────────────────────────────────────────────────────
    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Tải toàn bộ file âm thanh, tự động resample và downmix về mono"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            return y, sr
        except Exception as e:
            print(f"[load] Lỗi {audio_path}: {e}")
            return None, None

    # ── 2. Trim silence ───────────────────────────────────────────────────────
    def remove_silence(self, audio: np.ndarray, sr: int,
                       threshold_db: int = -40) -> np.ndarray:
        """Loại bỏ khoảng im lặng dựa trên năng lượng Mel-spectrogram"""
        S      = librosa.feature.melspectrogram(y=audio, sr=sr,
                                                hop_length=self.hop_length)
        S_db   = librosa.power_to_db(S, ref=np.max)
        energy = np.mean(S_db, axis=0)           # (T,)
        mask   = energy > threshold_db

        if mask.any():
            sample_mask = np.repeat(mask, self.hop_length)[:len(audio)]
            audio = audio[sample_mask]
        return audio

    # ── 3. Normalize ──────────────────────────────────────────────────────────
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Peak normalization về [-1, 1]"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    # ── 4. Extract features ───────────────────────────────────────────────────
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Trích xuất 4 đặc trưng, mỗi đặc trưng được pool bằng mean + std
        → vector cố định 68 chiều, không cần cắt/pad file

        Cấu trúc vector:
            MFCC        : mean(20) + std(20) = 40 chiều
            Chroma      : mean(12) + std(12) = 24 chiều
            ZCR         : mean( 1) + std( 1) =  2 chiều
            Spec.Centroid: mean( 1) + std( 1) =  2 chiều
            ─────────────────────────────────────────────
            Tổng        :                       68 chiều
        """
        def pool(x: np.ndarray) -> np.ndarray:
            return np.concatenate([np.mean(x, axis=1),
                                   np.std(x,  axis=1)])

        mfcc    = librosa.feature.mfcc(y=audio, sr=sr,
                                       n_mfcc=self.n_mfcc,
                                       hop_length=self.hop_length)        # (20, T)
        print(f"mfcc vector: {mfcc}")

        chroma  = librosa.feature.chroma_stft(y=audio, sr=sr,
                                              hop_length=self.hop_length) # (12, T)
        print(f"chroma vector: {chroma}")
        zcr     = librosa.feature.zero_crossing_rate(audio,
                                                     hop_length=self.hop_length) # (1, T)
        print(f"zcr vector: {zcr}")
        spec_c  = librosa.feature.spectral_centroid(y=audio, sr=sr,
                                                    hop_length=self.hop_length)  # (1, T)
        print(f"spec_c vector: {spec_c}")

        return np.concatenate([pool(mfcc), pool(chroma),
                                pool(zcr),  pool(spec_c)])                # (68,)

    # ── 5. Pipeline ───────────────────────────────────────────────────────────
    def preprocess(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Pipeline đầy đủ: load → trim → normalize → extract
        Trả về vector 68 chiều hoặc None nếu file lỗi
        """
        y, sr = self.load_audio(audio_path)
        if y is None:
            return None

        y = self.remove_silence(y, sr)
        if len(y) == 0:
            print(f"[preprocess] File toàn silence: {audio_path}")
            return None

        y = self.normalize_audio(y)
        return self.extract_features(y, sr)



if __name__ == "__main__":
    #test 
    processor = AudioPreprocessor()
    features = processor.preprocess(r"D:\Project\HCSDLDPT\wind_instruments_1s\0b82b3a5.wav")
    if features is not None:
        print("Đặc trưng đã được trích xuất thành công:")
        print(features.shape)
        print(features)

