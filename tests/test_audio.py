import logging

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.io import wavfile

from tests.xdf_writer import SyntheticStream, write_xdf
from xdf_tools.audio import (XDFAudioStream, extract_audio_stream_channels,
                             to_unit_range, xdf_to_wav)
from xdf_tools.file import XDFFile
from xdf_tools.stream import XDFStream

SRATE = 1000.0


def audio_stream(data, channel_format, srate=SRATE, start=100.0):
    data = np.asarray(data)
    return XDFStream({
        "info": {
            "name": ["audio"],
            "type": ["Audio"],
            "hostname": ["h"],
            "nominal_srate": [str(srate)],
            "channel_format": [channel_format],
        },
        "time_series": data,
        "time_stamps": start + np.arange(len(data)) / srate,
    })


def test_to_unit_range_int16_uses_full_scale():
    data, description = to_unit_range(np.array([-32768, 0, 16384, 32767], dtype=np.int16))
    np.testing.assert_allclose(data, [-1.0, 0.0, 0.5, 32767 / 32768])
    assert "scaled" in description


def test_to_unit_range_float_in_range_is_unchanged():
    data, _ = to_unit_range(np.array([-0.5, 0.25], dtype=np.float32))
    np.testing.assert_allclose(data, [-0.5, 0.25])


def test_to_unit_range_float_out_of_range_is_peak_scaled_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="xdf_tools.audio"):
        data, _ = to_unit_range(np.array([-4.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(data, [-1.0, 0.5])
    assert "outside the expected [-1, 1] range" in caplog.text


def test_extract_keeps_recorded_level_by_default():
    quiet = (np.sin(np.linspace(0, 20, 500)) * 1000).astype(np.int16)
    channels, fs = extract_audio_stream_channels(audio_stream(np.stack([quiet, quiet // 2], axis=1), "int16"))
    assert fs == SRATE
    assert len(channels) == 2 and all(c.dtype == np.int16 for c in channels)
    assert np.max(np.abs(channels[0])) == pytest.approx(1000, abs=1)
    assert np.max(np.abs(channels[1])) == pytest.approx(500, abs=1)


def test_extract_normalize_peak_normalizes_each_channel_independently():
    quiet = (np.sin(np.linspace(0, 20, 500)) * 1000).astype(np.int16)
    channels, _ = extract_audio_stream_channels(
        audio_stream(np.stack([quiet, quiet // 2], axis=1), "int16"), normalize=True)
    assert [int(np.max(np.abs(c))) for c in channels] == [32767, 32767]


def test_extract_rounds_rather_than_truncates():
    """0.00003 * 32767 = 0.98 and -0.00003 * 32767 = -0.98: rounding gives +-1, truncation would give 0."""
    samples = np.array([[0.00003], [-0.00003], [0.5]], dtype=np.float32)
    (channel,), _ = extract_audio_stream_channels(audio_stream(samples, "float32"))
    assert channel.tolist() == [1, -1, 16384]


def test_extract_clips_to_int16_range():
    samples = np.array([[1.0], [-1.0]], dtype=np.float32)
    (channel,), _ = extract_audio_stream_channels(audio_stream(samples, "float32"))
    assert channel.tolist() == [32767, -32767]


def test_extract_transposed_input_is_corrected():
    samples = np.zeros((2, 100), dtype=np.int16)
    channels, _ = extract_audio_stream_channels(audio_stream(samples, "int16"))
    assert len(channels) == 2 and len(channels[0]) == 100


def test_extract_rejects_nan_and_non_numeric():
    with pytest.raises(ValueError, match="NaN"):
        extract_audio_stream_channels(audio_stream(np.array([[0.1], [np.nan]], dtype=np.float32), "float32"))
    with pytest.raises(ValueError, match="non-numeric"):
        extract_audio_stream_channels(audio_stream(np.array([["a"], ["b"]]), "string"))


def test_xdf_to_wav_multichannel_and_separate(recording, tmp_path):
    out = tmp_path / "out" / "audio.wav"
    xdf_to_wav(recording, "audio", str(out))
    fs, samples = wavfile.read(out)
    assert fs == 2000 and samples.shape == (40000, 2) and samples.dtype == np.int16
    assert np.max(np.abs(samples)) == pytest.approx(10000, abs=2)

    separate = tmp_path / "separate" / "audio"
    xdf_to_wav(recording, "audio", str(separate), separate_channels=True, normalize=True)
    for n in (1, 2):
        fs, mono = wavfile.read(tmp_path / "separate" / f"audio_channel{n}.wav")
        assert mono.shape == (40000,) and np.max(np.abs(mono)) == 32767


@pytest.fixture
def audio(recording):
    return XDFFile(recording).stream_by_name("audio", stream_class=XDFAudioStream)


def test_audio_stream_class_and_properties(audio):
    assert isinstance(audio, XDFAudioStream)
    assert audio.sample_rate == 2000
    assert audio.samples.shape == (40000, 2) and audio.samples.dtype == np.int16
    assert audio._clocks_synced


def test_cut_returns_the_requested_window(audio):
    segment = audio.cut(audio.start_time + 1.0, 0.5)
    assert segment.sample_rate == 2000
    assert segment.samples.shape == (1000, 2)
    np.testing.assert_array_equal(segment.samples, audio.samples[2000:3000])


def test_cut_pads_with_silence_and_warns(audio, caplog):
    with caplog.at_level(logging.WARNING, logger="xdf_tools.audio"):
        segment = audio.cut(audio.start_time - 0.1, 0.3)
    assert segment.samples.shape == (600, 2)
    assert not segment.samples[:200].any()
    np.testing.assert_array_equal(segment.samples[200:], audio.samples[:400])
    assert "padding 0.100s of silence at the start" in caplog.text


def test_cut_mono_mix(audio):
    segment = audio.cut(audio.start_time + 1.0, 0.5, mix="mono")
    assert segment.samples.shape == (1000, 1)
    expected = np.round(audio.samples[2000:3000].astype(np.float64).mean(axis=1))
    np.testing.assert_array_equal(segment.samples[:, 0], expected.astype(np.int16))


def test_cut_rejects_unknown_mix(audio):
    with pytest.raises(ValueError, match="Unknown audio mix"):
        audio.cut(audio.start_time, 1.0, mix="surround")


def test_cut_with_zero_time_correction_matches_plain_cut(audio):
    plain = audio.cut(audio.start_time + 1.0, 0.5)
    corrected = audio.cut(audio.start_time + 1.0, 0.5, time_correction=lambda tau: np.zeros_like(tau))
    assert np.max(np.abs(corrected.samples.astype(int) - plain.samples.astype(int))) <= 1


def test_cut_with_time_correction_shifts_the_audio(audio):
    shift = 0.1  # seconds, i.e. 200 samples
    corrected = audio.cut(audio.start_time + 1.0, 0.5, time_correction=lambda tau: np.full_like(tau, shift))
    expected = audio.samples[2200:3200]
    assert np.max(np.abs(corrected.samples.astype(int) - expected.astype(int))) <= 1


def test_mono_window(audio):
    window = audio.mono_window(audio.start_time + 1.0, 1.0, rate=1000)
    assert window.shape == (1000,)
    assert audio.mono_window(audio.start_time - 5.0, 1.0, rate=1000) is None
    assert audio.mono_window(audio.start_time + 19.9, 1.0, rate=1000) is None


def test_gap_in_audio_timestamps_is_warned_about(tmp_path, caplog):
    time_stamps = np.concatenate([np.arange(0, 2, 0.001), np.arange(5, 7, 0.001)]) + 100.0
    stream = SyntheticStream(
        name="audio", type="Audio", hostname="h", nominal_srate=1000.0, channel_format="int16",
        time_stamps=time_stamps, data=np.zeros((len(time_stamps), 1), dtype=np.int16),
        clock_times=np.array([100.0, 107.0]), clock_values=np.array([0.0, 0.0]),
    )
    path = tmp_path / "gap.xdf"
    write_xdf(path, [stream])
    with caplog.at_level(logging.WARNING, logger="xdf_tools.audio"):
        XDFFile(str(path), dejitter_timestamps=False).stream_by_name("audio", stream_class=XDFAudioStream)
    assert "1 gap(s) longer than 1s" in caplog.text
