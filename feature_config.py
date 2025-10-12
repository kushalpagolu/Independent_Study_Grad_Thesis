NUM_CHANNELS = 14
SAMPLES_PER_SECOND = 256
NUM_BANDS = 5
NUM_HJORTH_PARAMS = 2

FEATURE_LENGTHS = {
    "band_power": NUM_CHANNELS * NUM_BANDS,                  # 70
    "hjorth": NUM_CHANNELS * NUM_HJORTH_PARAMS,              # 28
    "entropy": NUM_CHANNELS,                                 # 14
    "fractal": NUM_CHANNELS,                                 # 14
    "first_order": NUM_CHANNELS * SAMPLES_PER_SECOND,        # 3584
    "second_order": NUM_CHANNELS * SAMPLES_PER_SECOND,       # 3584
    "eeg_filtered": NUM_CHANNELS * SAMPLES_PER_SECOND        # 3584
}

# Compute cumulative indices
FEATURE_INDICES = {}
offset = 0
for key, length in FEATURE_LENGTHS.items():
    FEATURE_INDICES[key] = (offset, offset + length)
    offset += length

TOTAL_FEATURES = offset  # Should be 10702
