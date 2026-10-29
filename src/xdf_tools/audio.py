"""Audio-stream utilities: channel extraction to int16, WAV export, and XDFAudioStream."""
from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

try:
    from scipy.io.wavfile import write as write_wav
    from scipy.signal import resample_poly
except ImportError as exc:  # pragma: no cover
    raise ImportError("xdf_tools.audio requires scipy; install it with: pip install 'xdf-tools[audio]'") from exc

from xdf_tools.file import XDFFile
from xdf_tools.stream import XDFStream

logger = logging.getLogger(__name__)

# LSL streams can pause/resume; a jump in the timestamps larger than this means samples are missing
AUDIO_GAP_THRESHOLD_SECONDS = 1.0
# Samples are interpolated in blocks of this many to keep memory use bounded on long recordings
CORRECTED_CUT_BLOCK = 1 << 20


def to_unit_range(samples: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert raw XDF samples to float64 in [-1, 1], based on their actual dtype.

    Integer streams (e.g. int16 from a 16-bit capture) are scaled by their full-scale
    value. Float streams are assumed to already be in [-1, 1], as is the convention for
    audio; if they exceed it they are peak-scaled to fit (with a warning), since
    writing them as-is would clip. Returns (data, description of what was done).
    """
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        # Symmetric scale so that -full_scale maps to -1.0 (e.g. int16: 32768)
        full_scale = float(-info.min) if info.min < 0 else float(info.max)
        return samples.astype(np.float64) / full_scale, f"{samples.dtype} scaled by 1/{full_scale:g}"

    data = samples.astype(np.float64)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        logger.warning(
            f"{samples.dtype} audio peaks at {peak:g}, outside the expected [-1, 1] range; "
            f"scaling to fit to avoid clipping. Absolute levels are not preserved."
        )
        return data / peak, f"{samples.dtype} peak-scaled from {peak:g}"
    return data, f"{samples.dtype} used as-is (already within [-1, 1])"


def extract_audio_stream_channels(
        audio_stream: XDFStream,
        normalize: bool = False,
        ) -> tuple[list[np.ndarray], float]:
    """Extract 16-bit channel samples from a single audio stream.

    Samples are converted according to their stored dtype, so the
    result keeps the recorded volume levels.
    Set `normalize` to True to additionally peak-normalize
    each channel independently to full scale (e.g. to boost quiet recordings).

    Returns (list of one int16 array per channel, sampling rate).
    """
    # pyxdf returns time_series in the stream's declared channel_format dtype
    raw = audio_stream.time_series
    if raw.dtype == object or raw.dtype.kind in "US":
        raise ValueError(
            f"Audio stream has non-numeric channel_format "
            f"{audio_stream.channel_format!r}; cannot convert to WAV."
        )
    fs = audio_stream.nominal_srate

    # Ensure shape is (samples, channels)
    if raw.ndim == 1:
        raw = raw[:, None]  # mono -> (N,1)
    elif raw.shape[0] < raw.shape[1]:
        # likely (channels, samples) -> transpose
        # NB: audio streams typically have no per-channel label metadata, so
        # XDFStream.time_series can't disambiguate orientation on its own here
        raw = raw.T

    logger.info(f"Sampling rate: {fs}")
    channels = []
    for ch in range(raw.shape[1]):
        raw_channel = raw[:, ch]
        if np.issubdtype(raw_channel.dtype, np.floating):
            n_nan = np.isnan(raw_channel).sum()
            n_inf = np.isinf(raw_channel).sum()
            if n_nan or n_inf:
                raise ValueError(
                    f"Channel {ch} contains {n_nan} NaN and {n_inf} Inf samples "
                    f"out of {raw_channel.shape[0]} ({(n_nan + n_inf) / raw_channel.shape[0]:.1%}); "
                    f"audio stream data appears corrupted."
                )
        channel, how = to_unit_range(raw_channel)
        logger.info(f"Channel {ch}: {how}")
        if normalize:
            peak = np.max(np.abs(channel))
            if peak > 0:
                channel = channel / peak
            logger.info(f"Channel {ch}: peak-normalized (original peak {peak:.4g} of full scale)")
        else:
            logger.info(f"Channel {ch}: peak level {np.max(np.abs(channel)):.4g} of full scale")
        channels.append(np.round(np.clip(channel, -1.0, 1.0) * 32767).astype(np.int16))

    return channels, fs


def xdf_to_wav(
        xdf_file: str,
        audio_stream_label: str,
        wav_outfile: str,
        separate_channels: bool = False,
        normalize: bool = False,
        ) -> None:
    """Export an XDF file's audio stream to WAV: one multi-channel file, or one file per channel
    (`<name>_channel1.wav`, ...) if `separate_channels` is set."""
    audio_stream = XDFFile(xdf_file).stream_by_name(audio_stream_label)
    channel_samples, fs = extract_audio_stream_channels(audio_stream, normalize=normalize)

    if not wav_outfile.endswith(".wav"):
        wav_outfile += ".wav"
    os.makedirs(os.path.dirname(os.path.abspath(wav_outfile)), exist_ok=True)

    if separate_channels and len(channel_samples) > 1:
        for channel_n, channel_sample in enumerate(channel_samples):
            channel_n_outwav = wav_outfile[:-len(".wav")] + f"_channel{channel_n + 1}.wav"
            write_wav(channel_n_outwav, int(fs), channel_sample)
            logger.info(f"Audio stream (channel {channel_n + 1}) saved to {os.path.abspath(channel_n_outwav)}")
    else:
        multi_channel_audio = np.stack(channel_samples, axis=-1) if len(channel_samples) > 1 else channel_samples[0]
        write_wav(wav_outfile, int(fs), multi_channel_audio)
        logger.info(f"Audio stream saved to {os.path.abspath(wav_outfile)}")


@dataclass(frozen=True)
class AudioSegment:
    samples: np.ndarray  # (n_samples, n_channels) int16
    sample_rate: int


class XDFAudioStream(XDFStream):
    """The audio stream of an XDF file, addressable by XDF time."""

    def __init__(self, stream: dict, clocks_synced: bool = True):
        super().__init__(stream, clocks_synced=clocks_synced)
        channels, fs = extract_audio_stream_channels(self)
        self.samples = np.stack(channels, axis=-1)  # (n_samples, n_channels) int16
        self.sample_rate = int(round(fs))

        gaps = np.flatnonzero(np.diff(self.time_stamps) > AUDIO_GAP_THRESHOLD_SECONDS)
        if len(gaps) > 0:
            logger.warning(
                f"Audio stream has {len(gaps)} gap(s) longer than {AUDIO_GAP_THRESHOLD_SECONDS:g}s in its timestamps "
                f"(first at XDF time {self.time_stamps[gaps[0]]:.3f})."
            )

    def mono_window(self, start_time: float, duration: float, rate: int) -> np.ndarray | None:
        """Channel average of an XDF-time window, resampled to `rate`; None if the audio does not cover it."""
        first = int(round(float(self.index_at(np.array([start_time]))[0])))
        n = int(round(duration * self.sample_rate))
        if first < 0 or first + n > len(self.samples):
            return None
        mono = self.samples[first:first + n].astype(np.float64).mean(axis=1)
        ratio = Fraction(rate, self.sample_rate).limit_denominator(1000)
        return resample_poly(mono, ratio.numerator, ratio.denominator)

    def cut(
            self,
            start_time: float,
            duration: float,
            mix: str = "stereo",
            time_correction: Callable[[np.ndarray], np.ndarray] | None = None,
            ) -> AudioSegment:
        """Cut `duration` seconds starting at XDF time `start_time`, padding what the audio doesn't cover with silence.

        `mix` is "stereo" (keep the recorded channels) or "mono" (average them). `time_correction`, if given,
        maps seconds since the start of the cut to a (slowly varying) number of seconds to add to the XDF time
        that is read there, which warps the audio to compensate for clock drift against another stream.
        """
        if mix not in ("stereo", "mono"):
            raise ValueError(f"Unknown audio mix <{mix}>; expected 'stereo' or 'mono'")
        n_wanted = int(round(duration * self.sample_rate))
        n_stream, n_channels = self.samples.shape
        segment = np.zeros((n_wanted, n_channels), dtype=np.int16)

        if time_correction is None:
            start_index = int(round(float(self.index_at(np.array([start_time]))[0])))
            first = max(start_index, 0)
            last = min(start_index + n_wanted, n_stream)
            leading = first - start_index
            n_available = max(last - first, 0)
            trailing = n_wanted - leading - n_available
            if n_available > 0:
                segment[leading:leading + n_available] = self.samples[first:last]
        else:
            seconds = np.arange(n_wanted, dtype=np.float64) / self.sample_rate
            covered = np.zeros(n_wanted, dtype=bool)
            for block_start in range(0, n_wanted, CORRECTED_CUT_BLOCK):
                block = slice(block_start, min(block_start + CORRECTED_CUT_BLOCK, n_wanted))
                tau = seconds[block]
                positions = self.index_at(start_time + tau + time_correction(tau))
                lower = np.floor(positions).astype(np.int64)
                fraction = (positions - lower)[:, None]
                valid = (lower >= 0) & (lower + 1 <= n_stream - 1)
                lower = np.clip(lower, 0, n_stream - 2)
                values = self.samples[lower] * (1.0 - fraction) + self.samples[lower + 1] * fraction
                values[~valid] = 0
                segment[block] = np.round(values).astype(np.int16)
                covered[block] = valid
            # Silence is only reported at the two ends; a corrected cut has no gaps in the middle
            leading = int(np.argmax(covered)) if covered.any() else n_wanted
            trailing = int(np.argmax(covered[::-1])) if covered.any() else 0

        if mix == "mono" and segment.shape[1] > 1:
            segment = np.round(segment.astype(np.float64).mean(axis=1, keepdims=True)).astype(np.int16)

        leading_seconds = leading / self.sample_rate
        trailing_seconds = trailing / self.sample_rate
        if leading_seconds > 0 or trailing_seconds > 0:
            logger.warning(
                f"Audio does not cover the whole requested window: padding {leading_seconds:.3f}s of silence at the start "
                f"and {trailing_seconds:.3f}s at the end."
            )
        return AudioSegment(samples=segment, sample_rate=self.sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WAV file from XDF audio stream")
    parser.add_argument("--input_xdf", nargs="+", required=True, help="Path to input XDF file(s)")
    parser.add_argument("--output_wav", nargs="+", default=None, help="Path to output WAV file(s)")
    parser.add_argument("--audio_label", default="audio", help="Name of audio stream in XDF file")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Peak-normalize each channel to full scale (default: keep the recorded level)"
    )
    parser.add_argument(
        "--merge_channels",
        action="store_true",
        help="Merge all channels to a single WAV file instead of writing each channel to a separate WAV file"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

    input_xdf_files = args.input_xdf
    output_wav_files = args.output_wav
    if output_wav_files is not None:
        assert len(input_xdf_files) == len(output_wav_files), "Numbers of input and output file paths do not match!"
    else:
        output_wav_files = [os.path.splitext(xdf_file)[0] + ".wav" for xdf_file in input_xdf_files]

    failed_files = []
    for input_xdf, output_wav in zip(input_xdf_files, output_wav_files):
        logger.info(f"Extracting audio from <{args.audio_label}> field from {input_xdf} to {output_wav} ...")
        try:
            xdf_to_wav(
                xdf_file=input_xdf,
                audio_stream_label=args.audio_label,
                wav_outfile=output_wav,
                separate_channels=not args.merge_channels,
                normalize=args.normalize,
            )
        except Exception as e:
            logger.error(f"Failed to extract audio from {input_xdf}: {e}")
            failed_files.append(input_xdf)

    if failed_files:
        logger.warning(f"Failed to extract audio from {len(failed_files)} file(s): {failed_files}")


if __name__ == "__main__":
    main()
