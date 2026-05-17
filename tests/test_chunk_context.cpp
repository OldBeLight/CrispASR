// test_chunk_context.cpp — unit tests for the energy-minima chunker
// (issue #89, refresh after the 617cd02 context+filter revert).
//
// Pins audio_chunking::split_at_energy_minima:
//   - short audio → single slice
//   - long audio → contiguous, non-overlapping slices that cover the input
//   - all-zero (silence) input still partitions cleanly
//   - audio length == max_chunk → single slice
//
// The ±2 s slice context expansion + word-level trim that 617cd02 added
// was reverted because the encoder timestamp drift it required is not
// stable for parakeet TDT (issue #89 round 2 — words at slice boundaries
// landed outside both adjacent slice ranges and were silently dropped,
// and the text rebuild inserted a space between every kana on the JA
// tokenizer). The tests that pinned those code paths are gone with the
// code. All tests below are pure CPU, no model load.

#include <catch2/catch_test_macros.hpp>

#include "../src/core/audio_chunking.h"

#include <cstddef>
#include <vector>

TEST_CASE("chunking: short audio → single slice covering everything", "[unit][chunk-context]") {
    std::vector<float> audio(1600, 0.5f); // 0.1 s @ 16 kHz
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), audio.size(),
                                                         /*max_chunk=*/16000,
                                                         /*search_win=*/8000);
    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].first == 0);
    REQUIRE(slices[0].second == audio.size());
}

TEST_CASE("chunking: slices cover the full input without gaps or overlap", "[unit][chunk-context]") {
    // 4 s of uniform audio @ 16 kHz, split into 1 s chunks.
    const size_t SR = 16000;
    const size_t total = 4 * SR;
    std::vector<float> audio(total, 0.3f);
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), total,
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    // At least 2 slices for 4 s / 1 s.
    REQUIRE(slices.size() >= 2);

    // Contiguous: end[i] == begin[i+1].
    for (size_t i = 1; i < slices.size(); ++i) {
        REQUIRE(slices[i].first == slices[i - 1].second);
    }
    // First slice starts at 0, last slice ends at total.
    REQUIRE(slices.front().first == 0);
    REQUIRE(slices.back().second == total);
}

TEST_CASE("chunking: all-zero audio splits into equal chunks", "[unit][chunk-context]") {
    const size_t SR = 16000;
    const size_t total = 3 * SR;
    std::vector<float> audio(total, 0.0f); // pure silence → energy minima everywhere
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), total,
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    REQUIRE(slices.size() >= 2);
    REQUIRE(slices.front().first == 0);
    REQUIRE(slices.back().second == total);
    for (size_t i = 1; i < slices.size(); ++i)
        REQUIRE(slices[i].first == slices[i - 1].second);
}

TEST_CASE("chunking: audio exactly equal to max_chunk → single slice", "[unit][chunk-context]") {
    const size_t SR = 16000;
    std::vector<float> audio(SR, 0.2f);
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), audio.size(),
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].first == 0);
    REQUIRE(slices[0].second == SR);
}
