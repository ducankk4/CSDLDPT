from __future__ import annotations

from typing import Optional, Tuple

import librosa
import numpy as np


class AudioPreprocessor:
    """Tiền xử lý và trích xuất đặc trưng file âm thanh cho bài toán tìm kiếm.

    Tối ưu cho nhạc cụ hơi đơn âm (sáo, kèn, ...) · độ dài 2–30s.
    Output vector: ~68 chiều, L2-normalized → dùng cosine / dot-product similarity.
    """

    def __init__(
        self,
        sr: int = 22050,
        n_mfcc: int = 20,
        hop_length: int = 512,
        pre_emphasis: float = 0.97,
    ):
        """
        Args:
            sr            : Sample rate
            n_mfcc        : Số hệ số MFCC
            hop_length    : Bước nhảy frame (samples)
            pre_emphasis  : Hệ số pre-emphasis filter (0 = tắt)
        """
        self.sr            = sr
        self.n_mfcc        = n_mfcc
        self.hop_length    = hop_length
        self.pre_emphasis  = pre_emphasis

    # ── 1. Load ──────────────────────────────────────────────────────────────
    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Tải toàn bộ file âm thanh, tự động resample và downmix về mono."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            return y, sr
        except Exception as e:
            print(f"[load] Lỗi {audio_path}: {e}")
            return None, None

    # ── 2. Trim silence ───────────────────────────────────────────────────────
    def remove_silence(
        self,
        audio: np.ndarray,
        sr: int,
        top_db: int = 25,
        min_length_sec: float = 0.5,
    ) -> np.ndarray:
        """Cắt silence đầu/cuối, bảo toàn attack và release của nhạc cụ.

        Dùng librosa.effects.trim (top_db=25) thay vì Mel-energy threshold
        để tránh cắt nhầm phần attack quan trọng của nhạc cụ hơi.
        Nếu sau trim quá ngắn hơn min_length_sec → trả về bản gốc.
        """
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=2048,
            hop_length=self.hop_length,
        )
        min_samples = int(min_length_sec * sr)
        if len(trimmed) < min_samples:
            return audio  # trim quá mạnh → giữ nguyên
        return trimmed

    # ── 3. Normalize ──────────────────────────────────────────────────────────
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Peak normalization về [-1, 1].

        Không dùng StandardScaler vì nó thay đổi phân phối biên độ giữa
        các file, làm mất thông tin tương đối về dynamic range.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    # ── 4. Extract features ───────────────────────────────────────────────────
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trích xuất đặc trưng, pool bằng mean + std → vector cố định ~68 chiều.

        Cấu trúc vector (sau L2-normalize):
            MFCC              : mean(20) + std(20)          = 40 chiều
            Spec. Contrast    : mean( 7) + std( 7)          = 14 chiều
            F0 (pyin)         : mean + std + range          =  3 chiều
            Harmonic ratio    : scalar                      =  1 chiều
            ──────────────────────────────────────────────────────────
            Tổng                                            = 58 chiều

        """
        # ── helpers ──────────────────────────────────────────────────────────
        def pool2d(x: np.ndarray) -> np.ndarray:
            """mean + std theo trục thời gian cho ma trận (n_feat, T)."""
            return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])

        def pool1d(x: np.ndarray) -> np.ndarray:
            """mean + std cho vector (T,) hoặc (1, T)."""
            x = x.ravel()
            return np.array([np.mean(x), np.std(x)])

        # ── pre-emphasis: tăng cường harmonics tần số cao ────────────────────
        if self.pre_emphasis > 0:
            audio = np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

        # ── đặc trưng giữ lại ────────────────────────────────────────────────
        mfcc   = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
        )                                                           # (20, T)

        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, hop_length=self.hop_length
        )                                                           # ( 7, T)

        # ── F0 / cao độ (quan trọng nhất cho nhạc cụ đơn âm) ─────────────────
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz  – kèn bass thấp nhất
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz – sáo cao nhất
            sr=sr,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
        f0_feat   = np.array([
            np.mean(f0_voiced),             # cao độ trung bình
            np.std(f0_voiced),              # độ lệch cao độ (vibrato rộng?)
            np.ptp(f0_voiced),              # khoảng cao độ (range)
        ])                                                          # (3,)

        # # ── Harmonic ratio: tỉ lệ năng lượng harmonic / percussive ───────────
        harm, perc = librosa.effects.hpss(audio)
        h_ratio    = np.array([
            np.sum(harm ** 2) / (np.sum(perc ** 2) + 1e-8)
        ])                                                          # (1,)

        # ── Concatenate ───────────────────────────────────────────────────────
        feat = np.concatenate([
            pool2d(mfcc),       # 40d
            pool2d(contrast),   # 14d
            f0_feat,            #  3d
            h_ratio,            #  1d
        ])                      # = 58d

        # ── L2 normalize → cosine similarity = dot product ───────────────────
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    # ── 5. Pipeline ───────────────────────────────────────────────────────────
    def preprocess(self, audio_path: str) -> Optional[np.ndarray]:
        """Pipeline đầy đủ: load → trim → normalize → extract.

        Trả về vector 68 chiều (L2-normalized) hoặc None nếu file lỗi.
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


# # ── Similarity search helper ─────────────────────────────────────────────────
# def search_similar(
#     query_vec: np.ndarray,
#     db_vecs: np.ndarray,
#     top_k: int = 10,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Tìm top-K âm thanh tương đồng dùng cosine similarity.

#     Vì vector đã L2-normalized, cosine similarity = dot product.

#     Args:
#         query_vec : vector truy vấn (68,)
#         db_vecs   : ma trận database (N, 68)
#         top_k     : số kết quả trả về

#     Returns:
#         indices : chỉ số top-K trong db_vecs
#         scores  : cosine similarity tương ứng (cao hơn = tương đồng hơn)
#     """
#     scores  = db_vecs @ query_vec                   # (N,) – nhanh với numpy
#     top_idx = np.argsort(scores)[::-1][:top_k]
#     return top_idx, scores[top_idx]