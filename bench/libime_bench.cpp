// Benchmark for libime (Fcitx5 pinyin engine): accuracy, selection actions, per-keystroke latency.
// Settings mirror fcitx5-chinese-addons defaults plus the fuzzy flags Inner, InnerShort and PartialFinal.
#include <libime/core/userlanguagemodel.h>
#include <libime/pinyin/pinyincontext.h>
#include <libime/pinyin/pinyindictionary.h>
#include <libime/pinyin/pinyinencoder.h>
#include <libime/pinyin/pinyinime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
static long rssKb() {
    std::ifstream f("/proc/self/status"); std::string l;
    while (std::getline(f, l)) if (l.rfind("VmRSS:", 0) == 0) return std::stol(l.substr(6));
    return -1;
}
struct Item { std::string py, hz; };
static std::vector<Item> load(const char *fn) {
    std::vector<Item> v; std::ifstream f(fn); std::string l;
    while (std::getline(f, l)) { auto t = l.find('\t'); if (t != std::string::npos) v.push_back({l.substr(0, t), l.substr(t + 1)}); }
    return v;
}
static double pct(std::vector<double> &v, double p) { return v[std::min(v.size() - 1, size_t(v.size() * p))]; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s lm sc.dict sentences.tsv words.tsv\n", argv[0]); return 1; }
    long r0 = rssKb(); auto t0 = Clock::now();
    libime::PinyinIME ime(std::make_unique<libime::PinyinDictionary>(), std::make_unique<libime::UserLanguageModel>(argv[1]));
    ime.dict()->load(libime::PinyinDictionary::SystemDict, argv[2], libime::PinyinDictFormat::Binary);
    ime.setNBest(2); ime.setBeamSize(20); ime.setFrameSize(10); ime.setPartialLongWordLimit(4);
    ime.setFuzzyFlags(libime::PinyinFuzzyFlags{libime::PinyinFuzzyFlag::VE_UE} | libime::PinyinFuzzyFlag::Inner |
                      libime::PinyinFuzzyFlag::InnerShort | libime::PinyinFuzzyFlag::PartialFinal);
    libime::PinyinContext c(&ime);
    printf("open_ms=%.2f rss_before_kb=%ld rss_after_kb=%ld\n", us(t0, Clock::now()) / 1e3, r0, rssKb());

    auto S = load(argv[3]), W = load(argv[4]);
    int exact = 0, in3 = 0, in10 = 0, stepsTotal = 0, fail = 0;
    for (auto &s : S) {
        c.clear(); c.type(s.py);
        const auto &cands = c.candidates(); std::string top = cands.empty() ? "" : cands[0].toString();
        int rank = -1;
        for (size_t i = 0; i < cands.size() && i < 10; i++) if (cands[i].toString() == s.hz) { rank = i; break; }
        exact += rank == 0; in3 += rank >= 0 && rank < 3; in10 += rank >= 0;
        // greedy selection: pick the whole sentence if correct, else the longest correct prefix word
        int steps = 0; bool done = rank == 0;
        while (!done && steps < 12) {
            std::string rest = s.hz.substr(std::min(s.hz.size(), c.selectedSentence().size()));
            const auto &cs = c.candidates(); int found = -1; size_t best = 0;
            for (size_t i = 0; i < cs.size() && i < 60; i++) {
                auto str = cs[i].toString();
                if (str == rest) { found = i; best = SIZE_MAX; break; }
                if (str.size() > best && rest.compare(0, str.size(), str) == 0) { best = str.size(); found = i; }
            }
            if (found < 0) break;
            c.select(found); steps++;
            if (c.selected()) done = c.selectedSentence() == s.hz;
            if (c.selected() && !done) break;
        }
        if (!done) fail++; else stepsTotal += steps + 1;
        printf("S\t%s\t%s\t%s\t%s\n", rank == 0 ? "OK" : "XX", s.py.c_str(), s.hz.c_str(), top.c_str());
    }
    int n = S.size();
    printf("sent_n=%d top1_exact=%.1f%% in_top3=%.1f%% in_top10=%.1f%% avg_actions=%.2f unreachable=%d\n", n,
           100.0 * exact / n, 100.0 * in3 / n, 100.0 * in10 / n, double(stepsTotal) / (n - fail), fail);

    int w1 = 0, w5 = 0;
    for (auto &w : W) {
        c.clear(); c.type(w.py); const auto &cs = c.candidates(); int rank = -1;
        for (size_t i = 0; i < cs.size() && i < 300; i++) if (cs[i].toString() == w.hz) { rank = i; break; }
        w1 += rank == 0; w5 += rank >= 0 && rank < 5;
        printf("W\t%s\t%s\trank=%d\ttop1=%s\n", w.py.c_str(), w.hz.c_str(), rank, cs.empty() ? "" : cs[0].toString().c_str());
    }
    printf("word_n=%zu top1=%.1f%% top5=%.1f%%\n", W.size(), 100.0 * w1 / W.size(), 100.0 * w5 / W.size());

    std::vector<double> ks, full; std::string sink;
    for (int it = 0; it < 30; it++) for (auto &s : S) {
        c.clear();
        for (char ch : s.py) {
            auto a = Clock::now(); c.type(std::string(1, ch)); const auto &cs = c.candidates();
            for (size_t i = 0; i < cs.size() && i < 10; i++) sink = cs[i].toString();
            ks.push_back(us(a, Clock::now()));
        }
        c.clear(); auto a = Clock::now(); c.type(s.py); const auto &cs = c.candidates();
        for (size_t i = 0; i < cs.size() && i < 10; i++) sink = cs[i].toString();
        full.push_back(us(a, Clock::now()));
    }
    std::sort(ks.begin(), ks.end()); std::sort(full.begin(), full.end());
    printf("keystroke_n=%zu p50_us=%.1f p95_us=%.1f p99_us=%.1f max_us=%.1f\n", ks.size(), pct(ks, .5), pct(ks, .95), pct(ks, .99), ks.back());
    printf("fullsent_n=%zu p50_us=%.1f p95_us=%.1f p99_us=%.1f max_us=%.1f\n", full.size(), pct(full, .5), pct(full, .95), pct(full, .99), full.back());
    printf("rss_end_kb=%ld\n", rssKb());
    return 0;
}
