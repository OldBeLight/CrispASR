// test-crispasr-cache.cpp — unit tests for crispasr_cache helpers.
//
// Covers pure filesystem operations (file_present, dir, ensure_cached_file
// happy path) without making any network requests so the suite stays fast
// and hermetic.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_cache.h"

#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
static std::string make_temp_dir() {
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    // buf ends with backslash; trim it for consistent path joining
    std::string base = buf;
    if (!base.empty() && (base.back() == '\\' || base.back() == '/'))
        base.pop_back();
    std::string dir = base + "/crispasr_unit_" + std::to_string(_getpid());
    _mkdir(dir.c_str());
    return dir;
}
static void remove_file(const std::string& path) {
    DeleteFileA(path.c_str());
}
static void remove_dir(const std::string& path) {
    _rmdir(path.c_str());
}
#else
#include <sys/stat.h>
#include <unistd.h>
static std::string make_temp_dir() {
    const char* env = std::getenv("CRISPASR_SCRATCH_DIR");
    std::string base = (env && *env) ? env : ".scratch";
    mkdir(base.c_str(), 0755);
    std::string pattern = base + "/crispasr_unit_XXXXXX";
    std::string writable = pattern;
    char* buf = writable.data();
    return mkdtemp(buf) ? std::string(buf) : base;
}
static void remove_file(const std::string& path) {
    ::unlink(path.c_str());
}
static void remove_dir(const std::string& path) {
    ::rmdir(path.c_str());
}
#endif

static void write_file(const std::string& path, const char* content) {
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fputs(content, f);
        fclose(f);
    }
}

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

// ─── file_present() ──────────────────────────────────────────────────────────

TEST_CASE("file_present: nonexistent path returns false", "[unit]") {
    REQUIRE_FALSE(crispasr_cache::file_present("/this/absolutely/does/not/exist.bin"));
}

TEST_CASE("file_present: empty file (0-byte zombie) returns false", "[unit]") {
    const std::string tmp = make_temp_dir() + "/empty.bin";
    write_file(tmp, "");
    REQUIRE_FALSE(crispasr_cache::file_present(tmp));
    remove_file(tmp);
}

TEST_CASE("file_present: non-empty file returns true", "[unit]") {
    const std::string tmp = make_temp_dir() + "/nonempty.bin";
    write_file(tmp, "fake model bytes");
    REQUIRE(crispasr_cache::file_present(tmp));
    remove_file(tmp);
}

// ─── dir() ───────────────────────────────────────────────────────────────────

TEST_CASE("dir: override path is returned unchanged", "[unit]") {
    const std::string base = make_temp_dir();
    const std::string override_dir = base + "/my_cache";

    const std::string result = crispasr_cache::dir(override_dir);
    REQUIRE(result == override_dir);
}

TEST_CASE("dir: override creates the leaf directory", "[unit]") {
    const std::string base = make_temp_dir();
    const std::string override_dir = base + "/my_cache2";

    crispasr_cache::dir(override_dir);

    struct stat st {};
    REQUIRE(stat(override_dir.c_str(), &st) == 0);
#ifndef _WIN32
    REQUIRE(S_ISDIR(st.st_mode));
#endif
}

TEST_CASE("dir: empty override returns a non-empty platform default", "[unit]") {
    REQUIRE_FALSE(crispasr_cache::dir("").empty());
}

TEST_CASE("dir: environment and CLI destination precedence", "[unit]") {
    const char* old_cache = std::getenv("CRISPASR_CACHE_DIR");
    const char* old_models = std::getenv("CRISPASR_MODELS_DIR");
    const std::string saved_cache = old_cache ? old_cache : "";
    const std::string saved_models = old_models ? old_models : "";
    const bool had_cache = old_cache != nullptr;
    const bool had_models = old_models != nullptr;

    const std::string base = make_temp_dir();
    const std::string cache_env = base + "/cache_env";
    const std::string models_env = base + "/models_env";
    const std::string cli_override = base + "/cli_override";

    set_env("CRISPASR_CACHE_DIR", cache_env.c_str());
    set_env("CRISPASR_MODELS_DIR", models_env.c_str());
    REQUIRE(crispasr_cache::dir("") == cache_env);
    REQUIRE(crispasr_cache::dir(cli_override) == cli_override);

    set_env("CRISPASR_CACHE_DIR", nullptr);
    REQUIRE(crispasr_cache::dir("") == models_env);

    set_env("CRISPASR_CACHE_DIR", had_cache ? saved_cache.c_str() : nullptr);
    set_env("CRISPASR_MODELS_DIR", had_models ? saved_models.c_str() : nullptr);

    remove_dir(cli_override);
    remove_dir(models_env);
    remove_dir(cache_env);
    remove_dir(base);
}

// ─── ensure_cached_file() ────────────────────────────────────────────────────

TEST_CASE("ensure_cached_file: returns existing file path without fetching", "[unit]") {
    const std::string cache_dir = make_temp_dir();
    const std::string filename = "model.bin";
    const std::string full_path = cache_dir + "/" + filename;

    // Pre-populate the cache so no download is triggered.
    write_file(full_path, "fake model data");

    const std::string result =
        crispasr_cache::ensure_cached_file(filename,
                                           "https://example.invalid/model.bin", // URL — must not be reached
                                           /*quiet=*/true, "test", cache_dir);

    REQUIRE(result == full_path);
    remove_file(full_path);
}
