// fst_main.cpp — FaStar C++ Inference Engine CLI
//
// Build (see CMakeLists.txt for the canonical build):
//   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
//
// Usage:
//   ./build/ds4_npu_engine --model deepseek_v4_dspark.fst --prompt "Hello" --tokens 128

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <unistd.h>
#include <sys/file.h>
#include <fcntl.h>
#include <cerrno>
#include <sstream>
#include <mutex>

#include "fst_engine.h"
#include "nlohmann/json.hpp"
#include "httplib.h"

// ── BPE Tokenizer ──────────────────────────────────────────────────────
class BPETokenizer {
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id2str_;
    std::vector<std::pair<std::string, std::string>> merges_;
    /* Reverse of byte_to_unicode: codepoint -> original byte.  The ByteLevel
     * decoder maps each byte to a codepoint (printable bytes to themselves,
     * control/extended bytes to 256+n); decode() must invert that to recover
     * the real UTF-8 bytes.  Without this, '支持' renders as 'æĶ¯æĮģ'. */
    std::unordered_map<int, unsigned char> rev_byte_map_;

    static std::string byte_to_unicode(unsigned char b) {
        static const std::string map = [] {
            bool assigned[256] = {};
            std::string m(256, '\0');
            for (int b = 33; b <= 126; b++) { m[b] = (char)b; assigned[b] = true; }
            for (int b = 161; b <= 172; b++) { m[b] = (char)b; assigned[b] = true; }
            for (int b = 174; b <= 255; b++) { m[b] = (char)b; assigned[b] = true; }
            int n = 0;
            for (int b = 0; b < 256; b++) {
                if (!assigned[b]) { m[b] = (char)(256 + n); n++; }
            }
            return m;
        }();
        return std::string(1, map[b]);
    }

    /* Same map but as a full codepoint (the char cast above truncates the
     * 256+n codepoints to 8 bits, which is wrong for the reverse direction). */
    static int byte_to_unicode_cp(unsigned char b) {
        static int m[256];
        static bool init = [] {
            bool assigned[256] = {};
            for (int b = 33; b <= 126; b++) { m[b] = b; assigned[b] = true; }
            for (int b = 161; b <= 172; b++) { m[b] = b; assigned[b] = true; }
            for (int b = 174; b <= 255; b++) { m[b] = b; assigned[b] = true; }
            int n = 0;
            for (int bb = 0; bb < 256; bb++)
                if (!assigned[bb]) { m[bb] = 256 + n; n++; }
            return true;
        }();
        (void)init;
        return m[b];
    }

    static bool is_letter(uint32_t c) {
        if (c >= 'A' && c <= 'Z') return true;
        if (c >= 'a' && c <= 'z') return true;
        if (c >= 0x4E00 && c <= 0x9FFF) return true;
        if (c >= 0x3040 && c <= 0x309F) return true;
        if (c >= 0x30A0 && c <= 0x30FF) return true;
        if (c >= 0xAC00 && c <= 0xD7AF) return true;
        if (c >= 0x00C0 && c <= 0x024F) return true;
        return false;
    }

    static bool is_digit(uint32_t c) { return c >= '0' && c <= '9'; }

    static bool is_punct(uint32_t c) {
        if (c >= '!' && c <= '/') return true;
        if (c >= ':' && c <= '@') return true;
        if (c >= '[' && c <= '`') return true;
        if (c >= '{' && c <= '~') return true;
        return false;
    }

    static bool is_symbol(uint32_t c) {
        return (c >= 0x00A1 && c <= 0x00A9) ||
               (c >= 0x00AB && c <= 0x00B6) ||
               (c >= 0x00B8 && c <= 0x00FF) ||
               (c >= 0x2010 && c <= 0x2027) ||
               (c >= 0x2030 && c <= 0x205E) ||
               (c >= 0x2190 && c <= 0x23FF) ||
               (c >= 0x25A0 && c <= 0x27BF) ||
               (c >= 0x2E80 && c <= 0x303F) ||
               (c >= 0xFE30 && c <= 0xFE4F);
    }

    static uint32_t utf8_next(const std::string& s, size_t& i) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp = c;
        int len = 1;
        if ((c & 0x80) == 0) { len = 1; cp = c; }
        else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        for (int j = 1; j < len && i + j < s.size(); j++)
            cp = (cp << 6) | ((unsigned char)s[i + j] & 0x3F);
        i += len;
        return cp;
    }

    std::vector<std::string> split_words(const std::string& text) const {
        std::vector<std::string> words;
        size_t i = 0;
        size_t len = text.size();

        while (i < len) {
            size_t start = i;
            uint32_t c = utf8_next(text, i);

            if (is_digit(c)) {
                int count = 1;
                while (count < 3 && i < len) {
                    size_t peek = i;
                    uint32_t nc = utf8_next(text, peek);
                    if (is_digit(nc)) { i = peek; count++; }
                    else break;
                }
                words.push_back(text.substr(start, i - start));
                continue;
            }

            if (c >= 0x4E00 && c <= 0x9FFF) {
                words.push_back(text.substr(start, i - start));
                continue;
            }

            if (is_punct(c)) {
                size_t j = i;
                bool has_letter = false;
                while (j < len) {
                    size_t peek = j;
                    uint32_t pc = utf8_next(text, peek);
                    if (is_letter(pc)) { j = peek; has_letter = true; }
                    else break;
                }
                if (has_letter) {
                    words.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }

            if (is_letter(c)) {
                while (i < len) {
                    size_t j = i;
                    uint32_t pc = utf8_next(text, j);
                    if (is_letter(pc)) i = j;
                    else break;
                }
                words.push_back(text.substr(start, i - start));
                continue;
            }

            if (is_punct(c) || is_symbol(c)) {
                while (i < len) {
                    size_t j = i;
                    uint32_t pc = utf8_next(text, j);
                    if (is_punct(pc) || is_symbol(pc)) i = j;
                    else break;
                }
                words.push_back(text.substr(start, i - start));
                continue;
            }

            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                while (i < len) {
                    size_t j = i;
                    uint32_t pc = utf8_next(text, j);
                    if (pc == ' ' || pc == '\t' || pc == '\n' || pc == '\r') i = j;
                    else break;
                }
                words.push_back(text.substr(start, i - start));
                continue;
            }

            words.push_back(text.substr(start, i - start));
        }
        return words;
    }

public:
    explicit BPETokenizer(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            fprintf(stderr, "  [tokenizer] cannot open: %s\n", path.c_str());
            return;
        }
        nlohmann::json j;
        f >> j;

        /* Parse defensively: tokenizer.json formats vary — DeepSeek stores
         * merges as space-joined STRINGS, Hunyuan as 2-element ARRAYS; some
         * omit added_tokens.  A parse hiccup must NOT kill the engine: the HF
         * tokenizers bridge (hf_encode) is authoritative for encoding, and
         * decode only needs id2str_.  Catch + warn so the run continues. */
        try {
            auto& vocab = j["model"]["vocab"];
            int max_id = 0;
            for (auto it = vocab.begin(); it != vocab.end(); ++it) {
                int id = it.value();
                vocab_[it.key()] = id;
                if (id >= (int)id2str_.size()) id2str_.resize(id + 1);
                id2str_[id] = it.key();
                if (id > max_id) max_id = id;
            }

            auto& merges = j["model"]["merges"];
            for (auto& m : merges) {
                std::string a, b;
                if (m.is_string()) {                 /* DeepSeek: "a b" */
                    std::string s = m.get<std::string>();
                    size_t sp = s.find(' ');
                    if (sp != std::string::npos) { a = s.substr(0, sp); b = s.substr(sp + 1); }
                } else if (m.is_array() && m.size() == 2) {   /* Hunyuan: ["a","b"] */
                    a = m[0].get<std::string>(); b = m[1].get<std::string>();
                } else continue;
                merges_.push_back({a, b});
            }

            /* added_tokens (special tokens past the base vocab, e.g. Hunyuan
             * ids 120000+): register in id2str_ so decode() covers the full
             * vocab, and in vocab_ so the lossy C++ encode fallback emits them
             * instead of byte-coding them to id 0. */
            size_t n_added = 0;
            if (j.contains("added_tokens")) {
                for (auto& at : j["added_tokens"]) {
                    if (!at.is_object()) continue;
                    int id = at.value("id", -1);
                    std::string content = at.value("content", std::string());
                    if (id < 0 || content.empty()) continue;
                    if (id >= (int)id2str_.size()) id2str_.resize(id + 1);
                    id2str_[id] = content;
                    vocab_[content] = id;
                    if (id > max_id) max_id = id;
                    ++n_added;
                }
            }

            fprintf(stderr, "  [tokenizer] loaded: %zu vocab, %zu merges, %zu "
                    "added, max_id=%d\n", vocab_.size(), merges_.size(), n_added, max_id);
        } catch (const std::exception& e) {
            fprintf(stderr, "  [tokenizer] WARN: partial parse (%s); encode "
                    "will use the HF bridge, decode may miss some ids\n", e.what());
        }

        /* Build the reverse byte_to_unicode map (codepoint -> original byte)
         * used by decode() to invert the ByteLevel encoding. */
        for (int b = 0; b < 256; b++)
            rev_byte_map_[byte_to_unicode_cp((unsigned char)b)] = (unsigned char)b;
    }

    /* Decode one token id to its real UTF-8 text by reversing the ByteLevel
     * byte map.  Codepoints in the map (0..323) are byte-mapped back; any
     * codepoint outside the map (e.g. the '｜'/'▁' in special tokens) is kept
     * literally so <BOS>/<EOS>/<pad> render verbatim. */
    std::string decode_one(int id) const {
        if (id < 0 || id >= (int)id2str_.size() || id2str_[id].empty())
            return std::string();
        const std::string& s = id2str_[id];
        std::string out;
        size_t i = 0;
        while (i < s.size()) {
            uint32_t cp = utf8_next(s, i);
            auto it = rev_byte_map_.find((int)cp);
            if (it != rev_byte_map_.end())
                out.push_back((char)it->second);
            else {
                /* literal codepoint outside the byte map: re-emit its UTF-8 */
                if (cp < 0x80) out.push_back((char)cp);
                else if (cp < 0x800) {
                    out.push_back((char)(0xC0 | (cp >> 6)));
                    out.push_back((char)(0x80 | (cp & 0x3F)));
                } else if (cp < 0x10000) {
                    out.push_back((char)(0xE0 | (cp >> 12)));
                    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back((char)(0x80 | (cp & 0x3F)));
                } else {
                    out.push_back((char)(0xF0 | (cp >> 18)));
                    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
                    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back((char)(0x80 | (cp & 0x3F)));
                }
            }
        }
        return out;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<std::string> words = split_words(text);

        std::vector<int> ids;

        for (auto& word : words) {
            if (word.empty()) continue;

            std::vector<std::string> chars;
            for (size_t i = 0; i < word.size(); i++)
                chars.push_back(byte_to_unicode((unsigned char)word[i]));

            while (chars.size() > 1) {
                int best_rank = (int)merges_.size();
                int best_idx = -1;
                for (size_t j = 0; j + 1 < chars.size(); j++) {
                    for (int r = 0; r < (int)merges_.size(); r++) {
                        if (merges_[r].first == chars[j] && merges_[r].second == chars[j + 1]) {
                            if (r < best_rank) { best_rank = r; best_idx = (int)j; }
                            break;
                        }
                    }
                }
                if (best_idx < 0) break;

                chars[best_idx] = chars[best_idx] + chars[best_idx + 1];
                chars.erase(chars.begin() + best_idx + 1);
            }

            for (auto& token : chars) {
                auto it = vocab_.find(token);
                if (it != vocab_.end())
                    ids.push_back(it->second);
                else
                    ids.push_back(0);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string out;
        for (int id : tokens) out += decode_one(id);
        return out;
    }
};

static void print_token(int token_id, const char* text, void* user) {
    auto* tok = (BPETokenizer*)user;
    std::string s = tok->decode({token_id});
    std::cout << s;
    std::cout.flush();
    (void)text;
}

/* Encode the prompt with the real HuggingFace `tokenizers` ByteLevel BPE via
 * the fst_tokenize.py bridge.  The C++ BPETokenizer::encode cannot reproduce
 * DeepSeek's GPT-4-style Unicode pre-tokenizer (no \p{L}/\p{N} in std::regex)
 * and drops the space that becomes the Ġ prefix the vocab keys on, so it
 * falls back to id 0 = <BOS> for any word it can't merge and corrupts the
 * prompt.  Encoding is once-per-prompt, so a subprocess is fine.  The prompt
 * rides in the FST_PROMPT env var so shell-meta chars survive verbatim. */
static std::vector<int> hf_encode(const std::string& tok_path,
                                  const std::string& script_path,
                                  const std::string& prompt) {
    std::vector<int> ids;
    if (tok_path.empty() || script_path.empty()) return ids;
    setenv("FST_PROMPT", prompt.c_str(), 1);
    std::string cmd = "python3 \"" + script_path + "\" \"" + tok_path + "\" 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return ids;
    char* line = nullptr;
    size_t cap = 0;
    if (getline(&line, &cap, p) > 0) {
        const char* s = line;
        while (*s) {
            while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
            if (!*s) break;
            char* end = nullptr;
            long v = strtol(s, &end, 10);
            if (end == s) break;
            ids.push_back((int)v);
            s = end;
        }
    }
    free(line);
    pclose(p);
    return ids;
}

/* Wrap a raw user message in the DeepSeek chat template (non-thinking mode).
 * The HF tokenizers bridge recognises every special token embedded here
 * (BOS / <｜User｜> / <｜Assistant｜> / "thinking-end newline"), so BOS is part
 * of the template and must NOT be prepended separately.  The trailing newline
 * (the thinking-block close) forces a direct non-reasoning answer. */
static std::string make_chat_prompt(const std::string& prompt) {
    return std::string("<｜begin▁of▁sentence｜><｜User｜>") + prompt +
           std::string("<｜Assistant｜>\n");
}

struct CLIArgs {
    std::string model = "deepseek_v4_dspark.fst";
    std::string model_dir;          /* --model-dir: where .fst + tokenizer live
                                      * (auto-download target; default = CWD). */
    std::string draft_model;
    std::string prompt;
    std::string tokenizer;          /* tokenizer.json (FST_TOKENIZER env fallback) */
    std::string tokenize_script;    /* fst_tokenize.py (FST_TOKENIZE_SCRIPT env) */
    std::string kernel_dir;         /* kernels dir (FST_KERNEL_DIR env) */
    int tokens = 128;
    float temp = 0.7f;
    float top_p = 0.9f;
    bool skip_prefill = false;
    bool no_sd = false;          /* --no-sd: force plain autoregressive generate
                                  * even when --draft_model is loaded (skip the
                                  * DSpark speculative-decoding verify loop). */
    bool interactive = false;    /* --interactive: use the multi-turn API
                                  * (reset_session + prefill_user + decode_step)
                                  * with a persistent KV cache. */
    bool serve = false;          /* --serve: start the HTTP web UI / chat server. */
    int port = 8080;             /* --port: server listen port. */
    std::string web_dir = "web"; /* --web-dir: directory holding index.html. */
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --prompt \"<text>\" [options]\n"
        "  --model <path>          .fst model file (default: deepseek_v4_dspark.fst)\n"
        "  --model-dir <path>      directory holding the .fst models + tokenizer\n"
        "                          (auto-download target; default: current directory)\n"
        "  --draft_model <path>    DSpark draft .fst for speculative decoding\n"
        "  --prompt <text>         prompt text\n"
        "  --tokens <n>            max tokens to generate (default 128)\n"
        "  --temp <f>              sampling temperature (default 0.7)\n"
        "  --top_p <f>             top-p sampling (default 0.9)\n"
        "  --tokenizer <path>      tokenizer.json (or env FST_TOKENIZER)\n"
        "  --tokenize_script <p>   fst_tokenize.py (or env FST_TOKENIZE_SCRIPT)\n"
        "  --kernel_dir <path>     kernels dir (or env FST_KERNEL_DIR)\n"
        "  --skip-prefill          skip prefill (continue from current KV state)\n"
        "  --no-sd                 disable speculative decoding\n"
        "  --interactive           use the multi-turn API (prefill_user + decode_step)\n"
        "                          with a persistent KV cache across turns\n"
        "  --serve                 start the HTTP web UI / chat server\n"
        "  --port <n>              server listen port (default 8080)\n"
        "  --web-dir <path>        directory holding index.html (default ./web)\n",
        prog);
}

static CLIArgs parse_args(int argc, char** argv) {
    CLIArgs a;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) { a.model = argv[++i]; }
        else if (arg == "--model-dir" && i + 1 < argc) { a.model_dir = argv[++i]; }
        else if (arg == "--draft_model" && i + 1 < argc) { a.draft_model = argv[++i]; }
        else if (arg == "--prompt" && i + 1 < argc) { a.prompt = argv[++i]; }
        else if (arg == "--tokens" && i + 1 < argc) { a.tokens = std::atoi(argv[++i]); }
        else if (arg == "--temp" && i + 1 < argc) { a.temp = std::atof(argv[++i]); }
        else if (arg == "--top_p" && i + 1 < argc) { a.top_p = std::atof(argv[++i]); }
        else if (arg == "--tokenizer" && i + 1 < argc) { a.tokenizer = argv[++i]; }
        else if (arg == "--tokenize_script" && i + 1 < argc) { a.tokenize_script = argv[++i]; }
        else if (arg == "--kernel_dir" && i + 1 < argc) { a.kernel_dir = argv[++i]; }
        else if (arg == "--skip-prefill") { a.skip_prefill = true; }
        else if (arg == "--no-sd")        { a.no_sd = true; }
        else if (arg == "--interactive")  { a.interactive = true; }
        else if (arg == "--serve")        { a.serve = true; }
        else if (arg == "--port" && i + 1 < argc) { a.port = std::atoi(argv[++i]); }
        else if (arg == "--web-dir" && i + 1 < argc) { a.web_dir = argv[++i]; }
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); exit(1); }
    }
    return a;
}

/* ── HTTP web UI / chat server ────────────────────────────────────────────
 * Reuses the already-constructed engine + tokenizer + HF encode bridge, so
 * there is no second copy of the model or path-resolution logic.  Generation
 * requests are serialized by gen_mutex (the NPU is single-instance).  Tokens
 * are streamed back to the browser as Server-Sent Events (text/event-stream);
 * the KV cache persists across /generate calls for multi-turn chat, and
 * /reset starts a fresh session. */
static void run_server(FSTEngine& engine, BPETokenizer& tokenizer,
                       const std::string& tok_path, const std::string& tok_script,
                       int port, const std::string& web_dir) {
    httplib::Server svr;
    static std::mutex gen_mutex;   /* serialize NPU access across requests */

    /* GET / — serve the chat UI. */
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::string path = web_dir + "/index.html";
        std::ifstream f(path, std::ios::binary);
        if (!f) { res.status = 404; res.set_content("index.html not found at " + path, "text/plain"); return; }
        std::stringstream ss; ss << f.rdbuf();
        res.set_content(ss.str(), "text/html; charset=utf-8");
    });

    /* POST /reset — clear KV cache + compressor, start a fresh chat. */
    svr.Post("/reset", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(gen_mutex);
        engine.reset_session();
        res.set_content("{\"ok\":true}", "application/json");
    });

    /* POST /generate  body: {prompt, tokens?, temp?, top_p?, reset?}
     *   -> SSE stream of {id, text, done} events. */
    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        nlohmann::json j = nlohmann::json::parse(req.body, nullptr, false);
        if (j.is_discarded() || !j.contains("prompt")) {
            res.status = 400; res.set_content("{\"error\":\"prompt required\"}", "application/json");
            return;
        }
        std::string prompt = j.value("prompt", "");
        int   tokens = j.value("tokens", 128);
        float temp   = j.value("temp", 0.7f);
        float top_p  = j.value("top_p", 0.9f);
        bool  reset  = j.value("reset", false);
        if (prompt.empty()) {
            res.status = 400; res.set_content("{\"error\":\"empty prompt\"}", "application/json");
            return;
        }

        std::string chat = make_chat_prompt(prompt);
        std::vector<int> ids = hf_encode(tok_path, tok_script, chat);
        if (ids.empty()) ids = tokenizer.encode(chat);

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, ids, tokens, temp, top_p, reset](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::lock_guard<std::mutex> lk(gen_mutex);
                if (reset) engine.reset_session();
                auto emit = [&](int tid, bool done) {
                    nlohmann::json ev = {
                        {"id", tid},
                        {"text", tokenizer.decode({tid})},
                        {"done", done}
                    };
                    std::string s = "data: " + ev.dump() + "\n\n";
                    sink.write(s.c_str(), s.size());
                };
                int tid = engine.prefill_user(ids, temp, top_p);
                if (tid >= 0) emit(tid, false);
                for (int i = 1; i < tokens && tid != engine.eos_token_id(); i++) {
                    tid = engine.decode_step(tid, temp, top_p);
                    emit(tid, false);
                    if (engine.seq_pos() >= engine.config_.max_seq - 1) break;
                }
                emit(engine.eos_token_id(), true);   /* terminal [DONE] marker */
                sink.done();
                return true;
            });
    });

    fprintf(stderr, "\n[serve] Web UI ready:  http://localhost:%d/   (web-dir: %s)\n",
            port, web_dir.c_str());
    fprintf(stderr, "[serve] POST /generate {prompt} -> SSE token stream;  POST /reset -> new session.\n");
    fflush(stderr);
    svr.listen("0.0.0.0", port);
}

/* ── HuggingFace model auto-download ───────────────────────────────────────
 * A fresh checkout ships no model weights and no tokenizer.  On first run,
 * FaStar fetches the missing artifacts from HuggingFace so the engine is
 * runnable without manual setup.  Downloads are resumable: a partial file
 * (wget -c / curl -L -C -) is continued by simply re-running. */
static constexpr const char* HF_REPO_URL =
    "https://huggingface.co/RaffaelloMolinari/Deepseek-V4-Flash-DSpark-FST/resolve/main/";
/* tokenizer.json is a standard DeepSeek file; if the FaStar repo does not yet
 * carry it, fall back to the upstream deepseek-ai model (always public). */
static constexpr const char* HF_TOKENIZER_FALLBACK_URL =
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/resolve/main/tokenizer.json";

static std::string join_dir(const std::string& dir, const std::string& fname) {
    return dir.empty() ? fname : (dir + "/" + fname);
}

/* Fetch url -> dest (resumable; leaves no junk file on HTTP error).  Returns
 * true if dest exists and is non-empty after the call. */
static bool fetch_to(const std::string& url, const std::string& dest) {
    std::string cmd = "wget -c --no-content-on-error \"" + url + "\" -O \"" + dest + "\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        fprintf(stderr, "  [download] wget failed (rc=%d); trying curl...\n", rc);
        std::remove(dest.c_str());
        cmd = "curl -fL -C - \"" + url + "\" -o \"" + dest + "\"";
        rc = std::system(cmd.c_str());
    }
    if (rc != 0 || access(dest.c_str(), F_OK) != 0) {
        std::remove(dest.c_str());   /* leave no junk file behind on failure */
        return false;
    }
    return true;
}

/* Download a single file from the FaStar HF repo into dir/fname if missing. */
static bool ensure_file(const std::string& dir, const std::string& fname) {
    std::string dest = join_dir(dir, fname);
    if (access(dest.c_str(), F_OK) == 0) return true;
    fprintf(stderr, "  [download] %s -> %s\n", fname.c_str(), dest.c_str());
    if (fetch_to(std::string(HF_REPO_URL) + fname, dest)) return true;
    fprintf(stderr, "  [download] FAILED: %s — is it present in the HF repo?\n", fname.c_str());
    return false;
}

/* tokenizer.json is tiny and always required; ensure it independently of the
 * big weights, with an upstream deepseek-ai fallback so a fresh clone can run
 * even before the FaStar repo is fully populated. */
static bool ensure_tokenizer(const std::string& model_dir) {
    std::string dest = join_dir(model_dir, "tokenizer.json");
    if (access(dest.c_str(), F_OK) == 0 || access("tokenizer.json", F_OK) == 0) return true;
    fprintf(stderr, "  [download] tokenizer.json -> %s\n", dest.c_str());
    if (fetch_to(std::string(HF_REPO_URL) + "tokenizer.json", dest)) return true;
    fprintf(stderr, "  [download] not in FaStar repo; falling back to deepseek-ai/DeepSeek-V4-Flash-DSpark...\n");
    if (fetch_to(HF_TOKENIZER_FALLBACK_URL, dest)) return true;
    fprintf(stderr, "  [download] FAILED: tokenizer.json\n");
    return false;
}

/* Ensure the big model weights + essential sidecars live in model_dir; if any
 * are missing, print a banner and download them.  The .fst.norm / .fst.hc
 * sidecars are required for coherent output (the bare .fst alone has
 * 40x-too-small RMSNorm weights).  When need_draft is set the DSpark draft
 * model is fetched too (only needed for speculative decoding).  tokenizer.json
 * is handled separately (tiny, always required, upstream fallback). */
static void ensure_model_files(const std::string& model_dir, bool need_draft) {
    const char* big_files[] = {
        "deepseek_v4_dspark.fst",
        "deepseek_v4_dspark.fst.hc",
        "deepseek_v4_dspark.fst.norm",
        "deepseek_v4_dspark.fst.tid2eid",
        nullptr
    };
    bool any_big_missing = false;
    for (int i = 0; big_files[i]; i++)
        if (access(join_dir(model_dir, big_files[i]).c_str(), F_OK) != 0) { any_big_missing = true; break; }
    if (need_draft && access(join_dir(model_dir, "dspark_draft.fst").c_str(), F_OK) != 0)
        any_big_missing = true;

    if (any_big_missing) {
        fprintf(stderr,
            "\nModel weights not found in %s. Downloading from HuggingFace\n"
            "  repo: RaffaelloMolinari/Deepseek-V4-Flash-DSpark-FST\n"
            "  (one-time download, ~163 GB; resumable — re-run to continue a partial fetch)\n\n",
            model_dir.empty() ? "the current directory" : model_dir.c_str());
        for (int i = 0; big_files[i]; i++) ensure_file(model_dir, big_files[i]);
        if (need_draft) ensure_file(model_dir, "dspark_draft.fst");
        fprintf(stderr, "\n[download] model weights acquisition complete.\n\n");
    }

    if (access(join_dir(model_dir, "tokenizer.json").c_str(), F_OK) != 0 &&
        access("tokenizer.json", F_OK) != 0) {
        fprintf(stderr, "\ntokenizer.json not found. Downloading from HuggingFace...\n\n");
        ensure_tokenizer(model_dir);
        fprintf(stderr, "\n[download] tokenizer acquisition complete.\n\n");
    }
}

int main(int argc, char** argv) {
    /* ── Single-instance lock: prevent multiple FaStar processes from
     *     fighting over the NPU hw_contexts (max 16 on AMDXDNA driver). */
    int lock_fd = ::open("/tmp/fastar_npu.lock", O_CREAT | O_RDWR, 0644);
    if (lock_fd < 0) {
        fprintf(stderr, "FATAL: Cannot create lock file /tmp/fastar_npu.lock: %s\n",
                strerror(errno));
        return 1;
    }
    if (flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            fprintf(stderr, "FATAL: Another FaStar instance is already running.\n");
        } else {
            fprintf(stderr, "FATAL: Cannot acquire lock: %s\n", strerror(errno));
        }
        ::close(lock_fd);
        return 1;
    }
    /* lock_fd stays open; OS releases on _exit() */

    CLIArgs args = parse_args(argc, argv);

    if (args.prompt.empty() && !args.serve) {
        fprintf(stderr, "Error: --prompt is required (or use --serve for the web UI).\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Resolve model_dir: --model-dir overrides; otherwise the model is expected
     * in the current directory.  If --model is a bare filename and a model_dir
     * was given, anchor it inside model_dir so the engine + auto-download agree
     * on where the weights live. */
    const std::string& model_dir = args.model_dir;
    if (!model_dir.empty() && args.model.find('/') == std::string::npos)
        args.model = model_dir + "/" + args.model;
    if (!model_dir.empty() && !args.draft_model.empty() && args.draft_model.find('/') == std::string::npos)
        args.draft_model = model_dir + "/" + args.draft_model;

    /* Auto-download the canonical model set from HuggingFace on first run.
     * Only applies to the default model name (a user-supplied custom --model is
     * assumed to be managed manually).  The draft is fetched only when SD is
     * actually requested (--draft_model). */
    {
        std::string base = args.model;
        size_t slash = base.find_last_of('/');
        if (slash != std::string::npos) base = base.substr(slash + 1);
        if (base == "deepseek_v4_dspark.fst")
            ensure_model_files(model_dir, !args.draft_model.empty());
    }

    fprintf(stderr, "FaStar C++ Engine\n");
    fprintf(stderr, "  model : %s\n", args.model.c_str());
    if (!args.draft_model.empty())
        fprintf(stderr, "  draft : %s\n", args.draft_model.c_str());
    fprintf(stderr, "  prompt: \"%s\"\n", args.prompt.c_str());
    fprintf(stderr, "  tokens: %d\n", args.tokens);
    fprintf(stderr, "  temp  : %.2f\n", args.temp);
    fprintf(stderr, "  top_p : %.2f\n", args.top_p);

    // ── Environment check ────────────────────────────────────────────────
    {
        const char* xrt_env = std::getenv("XILINX_XRT");
        const char* ld_path = std::getenv("LD_LIBRARY_PATH");
        bool has_xrt = xrt_env && xrt_env[0] != '\0';
        bool has_ld  = ld_path && std::string(ld_path).find("xrt") != std::string::npos;
        if (!has_xrt && !has_ld) {
            fprintf(stderr,
                "[WARN] XILINX_XRT not set and LD_LIBRARY_PATH has no XRT paths. "
                "NPU kernels may fail to load.\n");
        }
    }

    /* Resolve tokenizer.json: --tokenizer / FST_TOKENIZER, else look next to
     * the model (model_dir/tokenizer.json) and in the CWD.  The auto-downloader
     * places tokenizer.json in model_dir. */
    std::string tok_path = args.tokenizer;
    if (tok_path.empty()) {
        if (const char* e = std::getenv("FST_TOKENIZER")) tok_path = e;
    }
    BPETokenizer tokenizer("");
    if (tok_path.empty()) {
        std::string d_tok = join_dir(model_dir, "tokenizer.json");
        const std::string candidates[] = { d_tok, "tokenizer.json" };
        for (const auto& c : candidates)
            if (access(c.c_str(), F_OK) == 0) { tok_path = c; break; }
    }
    if (!tok_path.empty() && access(tok_path.c_str(), F_OK) == 0)
        tokenizer = BPETokenizer(tok_path);
    else
        fprintf(stderr, "[WARN] no tokenizer.json found (--tokenizer / FST_TOKENIZER / "
                        "model_dir); using lossy C++ fallback encoder.\n");

    /* Resolve the HF tokenize bridge script: --tokenize_script /
     * FST_TOKENIZE_SCRIPT, else ./scripts/fst_tokenize.py, then next to the
     * model (model_dir/fst_tokenize.py), then ./fst_tokenize.py (CWD). */
    std::string tok_script = args.tokenize_script;
    if (tok_script.empty()) {
        if (const char* e = std::getenv("FST_TOKENIZE_SCRIPT")) tok_script = e;
    }
    if (tok_script.empty()) {
        std::string d_script = join_dir(model_dir, "fst_tokenize.py");
        const std::string s_candidates[] = {
            "scripts/fst_tokenize.py", d_script, "fst_tokenize.py"
        };
        for (const auto& c : s_candidates)
            if (access(c.c_str(), F_OK) == 0) { tok_script = c; break; }
    }

    /* --kernel_dir / FST_KERNEL_DIR: where the _insts.bin / .xclbin live. */
    if (!args.kernel_dir.empty()) setenv("FST_KERNEL_DIR", args.kernel_dir.c_str(), 1);

    /* ExpertPager RAM cache.  Default 6 GB (a TRANSIENT SSD-load staging area
     * for the per-op path, where the authoritative expert cache is the
     * host_only BO cache in get_expert_bo, checked BEFORE the pager — keeping
     * the pager small avoids the 2x pager+BO duplication that OOM'd under a
     * cgroup cap).
     *
     * For the FUSED HY3 path (FST_HY3_FUSED_FFN), the FFN pack bypasses
     * get_expert_bo and reads directly from the pager's Expert structs, so the
     * pager is the ONLY expert cache — a 6 GB pager holds ~600 experts ≈ one
     * token (80×8=640), giving zero cross-token reuse.  FST_RAM_CACHE_GB raises
     * it (e.g. 20 GB ≈ 2000 experts ≈ 3 tokens) to capture cross-token reuse;
     * the fused path has no 2x BO duplication, so RSS stays ~40 GB on a 60 GB
     * box.  60 GB RAM, ~55 GB available — see HY3_FUSED_FFN_POSTMORTEM. */
    size_t ram_mb = 6000;
    if (const char *e = std::getenv("FST_RAM_CACHE_GB"))
        ram_mb = (size_t)std::strtoul(e, nullptr, 10) * 1024ULL;
    FSTEngine engine(args.model, ram_mb);

    fprintf(stderr, "\nModel configuration:\n");
    fprintf(stderr, "  Layers    : %d\n", engine.config_.n_layers);
    fprintf(stderr, "  Hidden Dim: %d\n", engine.config_.hidden_dim);
    fprintf(stderr, "  Experts   : %d\n", engine.config_.n_experts);
    fprintf(stderr, "  Top-K     : %d\n", engine.config_.top_k);

    /* Load draft model if specified */
    if (!args.draft_model.empty()) {
        fprintf(stderr, "\nLoading DSpark draft model: %s\n", args.draft_model.c_str());
        engine.load_draft_model(args.draft_model);
        fprintf(stderr, "  Draft Layers: %d\n", engine.draft_cfg_.n_layers);
        fprintf(stderr, "  Draft Block Size: %d (gamma=%d)\n",
                engine.draft_cfg_.block_size, engine.draft_cfg_.block_size - 1);
    }

    /* ── Web UI / chat server mode ─────────────────────────────────────
     * In --serve mode we skip the one-shot prompt; the server issues
     * prefill_user + decode_step per /generate request (multi-turn KV cache). */
    if (args.serve) {
        run_server(engine, tokenizer, tok_path, tok_script, args.port, args.web_dir);
        std::cout.flush();
        _exit(0);
    }

    /* Wrap the prompt in the official DeepSeek-V4 chat template before
     * encoding.  Matches ds4.c encode_chat_prompt() (line 22285) for
     * non-thinking ("chat") mode:
     *   BOS  <｜User｜> {prompt} <｜Assistant｜> <think/>
     * where <think/> is the think_end_id special token.  NOTE: the marker is
     * the CLOSING tag "</think>" (verified: the HF tokenizers bridge encodes
     * "</think>" -> single token 128822, identical to ds4.c's think_end_id).
     * The self-closing "<think/>" is NOT a special token — it byte-tokenizes
     * to [30, 37947, 33589] ("<","think","/>"), which would never enter
     * non-thinking mode and would inject three junk tokens.  The bridge
     * recognises every special token embedded in the string
     * (<｜begin▁of▁sentence｜> -> 0, <｜User｜> -> 128803,
     *  <｜Assistant｜> -> 128804, "</think>" -> 128822), so BOS is part of the
     * template and must NOT be prepended manually.  Without the trailing
     * "</think>" the model sits after <｜Assistant｜> in an undefined state
     * (it may emit a stray thinking token or drift); "</think>" forces a
     * direct non-reasoning answer, which is what an instruction-tuned
     * assistant should produce for a plain prompt. */
    /* HY3 (Hunyuan-3.0): apply the official no_think chat template (single user
     * turn, no tools) — BOS + <reasoning_mode> + "reasoning_effort:no_think" +
     * User + prompt + Assistant + empty <think></think>.  Mirrors HF's
     * apply_chat_template (chat_template.jinja); the HF bridge recognizes the
     * special tokens as their single IDs.  DeepSeek stays on its own template
     * (its special tokens are NOT in the HY3 vocab and vice-versa). */
    const bool hy3_raw = (engine.config_.arch == ARCH_HY3) && std::getenv("FST_HY3_RAW_PROMPT");
    const std::string chat_prompt =
        (engine.config_.arch == ARCH_HY3)
        ? (hy3_raw ? args.prompt
           : (std::string("<｜hy_begin_of_sentence:opensource｜>") +
              std::string("<｜reasoning_mode:opensource｜>reasoning_effort:no_think") +
              std::string("<｜hy_User:opensource｜>") + args.prompt +
              std::string("<｜hy_Assistant:opensource｜>") +
              std::string("<think:opensource></think:opensource>")))
        : std::string("<｜begin▁of▁sentence｜><｜User｜>") + args.prompt +
        std::string("<｜Assistant｜>\n");
    fprintf(stderr, "  [chat-template] %s (%zu bytes)\n",
            hy3_raw ? "RAW HY3 prompt (test bypass)"
                    : (engine.config_.arch == ARCH_HY3 ? "templated HY3 prompt" : "wrapped DS4 prompt"),
            chat_prompt.size());

    /* Encode via the HF tokenizers bridge (correct ByteLevel BPE).  Fall back
     * to the (lossy) C++ encoder only if the bridge is unavailable. */
    std::vector<int> prompt_ids = hf_encode(tok_path, tok_script, chat_prompt);
    if (prompt_ids.empty()) {
        fprintf(stderr, "  [tokenizer] HF bridge failed; using C++ fallback\n");
        prompt_ids = tokenizer.encode(chat_prompt);
    }

    fprintf(stderr, "  encoded %zu tokens: [", prompt_ids.size());
    for (size_t i = 0; i < prompt_ids.size() && i < 10; i++)
        fprintf(stderr, "%d%s", prompt_ids[i],
                i + 1 < std::min(prompt_ids.size(), (size_t)10) ? ", " : "");
    if (prompt_ids.size() > 10) fprintf(stderr, "...");
    fprintf(stderr, "]\n");

    if (args.interactive) {
        /* ── Interactive multi-turn path (persistent KV cache) ───────────
         * reset_session -> prefill_user (first reply token) -> decode_step
         * loop until EOS or the requested token budget.  Numerically
         * identical to generate()'s prefill+decode; the only difference is
         * the append-aware prefill (KV/compressor persist for the next turn). */
        fprintf(stderr, "\n[Interactive] Multi-turn API (persistent KV cache).\n");
        engine.reset_session();

        auto t0 = std::chrono::steady_clock::now();
        int tid = engine.prefill_user(prompt_ids, args.temp, args.top_p);
        auto t1 = std::chrono::steady_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "\n  Prefill Time: %.1f ms (%zu tokens)\n", prefill_ms, prompt_ids.size());

        if (tid >= 0) {
            std::string s = tokenizer.decode({tid});
            std::cout << s; std::cout.flush();
        }
        int generated = (tid >= 0) ? 1 : 0;
        int N = args.tokens > 0 ? args.tokens : 128;
        for (int i = 1; i < N && tid != engine.eos_token_id(); i++) {
            tid = engine.decode_step(tid, args.temp, args.top_p);
            std::string s = tokenizer.decode({tid});
            std::cout << s; std::cout.flush();
            generated++;
            if (engine.seq_pos() >= engine.config_.max_seq - 1) break;
        }
        auto t2 = std::chrono::steady_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double tps = (generated > 1) ? (double)(generated - 1) / (decode_ms / 1000.0) : 0.0;
        fprintf(stderr, "\n  Decode Tokens/sec: %.2f\n", tps);
    } else if (engine.has_draft() && !args.no_sd) {
        fprintf(stderr, "\n[DSpark] Starting speculative decoding with gamma=%d\n",
                engine.draft_cfg_.block_size - 1);
        engine.generate_dspark(prompt_ids, args.tokens, args.temp, args.top_p,
                                print_token, &tokenizer, args.skip_prefill);
    } else {
        if (args.no_sd)
            fprintf(stderr, "\n[Generate] --no-sd: forcing plain autoregressive decode (no draft/verify).\n");
        else
            fprintf(stderr, "\n[Generate] Plain (non-SD) path — validating main MLA.\n");
        engine.generate(prompt_ids, args.tokens, args.temp, args.top_p,
                        print_token, &tokenizer);
    }

    /* Bypass C++ destructors: tearing down 16 NPU hw_contexts + hundreds of
     * BOs can hang on the AMDXDMA driver on exit.  _exit() releases the OS
     * resources immediately (the reference fst_npu_executor does the same). */
    std::cout.flush();
    _exit(0);
}
