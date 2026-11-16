"""Turn technical measurements into selection penalties and resolution bands."""

import numpy as np
import pyarrow as pa


def numbers(table, name):
    if name not in table.column_names:
        return np.full(len(table), np.nan)
    field = table.schema.field(name)
    if name == "has_audio" and not pa.types.is_boolean(field.type):
        return np.full(len(table), np.nan)
    if not (pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or
            (name == "has_audio" and pa.types.is_boolean(field.type))):
        return np.full(len(table), np.nan)
    return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=float)


def measurements(table, config):
    width, height = numbers(table, "width"), numbers(table, "height")
    side = np.minimum(width, height)
    known_side = np.isfinite(width) & np.isfinite(height) & (width > 0) & (height > 0)
    metrics = {
        "short_side": (side, known_side, config["min_short_side"]),
        "fps": (numbers(table, "fps"), None, config["min_fps"]),
        "duration": (numbers(table, "duration_seconds"), None, config["min_duration_seconds"]),
        "audio_sample_rate": (numbers(table, "asr"), None, config["min_audio_sample_rate_hz"]),
    }
    failed, unknown = {}, {}
    for name, (value, known, minimum) in metrics.items():
        if known is None:
            known = np.isfinite(value) & (value > 0)
        good = known & (value >= minimum)
        if name == "audio_sample_rate" and "has_audio" in table.column_names:
            audio = numbers(table, "has_audio")
            known &= np.isfinite(audio)
            good &= audio == 1
        failed[name] = ~good
        unknown[name] = ~known
    signature = np.zeros(len(table), dtype=np.int32)
    for index, mask in enumerate(failed.values()):
        signature |= mask.astype(np.int32) << index
    thresholds = sorted(rule["min_short_side"] for rule in config["resolution_shares"])
    bands = np.where(known_side, np.searchsorted(thresholds, side, side="right"), 0)
    penalty = sum(mask.astype(float) for mask in failed.values())
    return bands, signature, penalty
