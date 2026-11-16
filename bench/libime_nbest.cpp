// Dump libime's full-sentence N-best (text, libime score) per test sentence as JSONL, for offline reranking.
// Engine settings match libime_bench.cpp except NBest (argv[4]).
// Input lines: typed \t ref [\t keys]. Without keys a full-sentence candidate spells exactly the typed pinyin; with
// keys (abbreviated typing, e.g. "br" -> keys "b'r", "zaosh" -> "zao'sh") it has one syllable per key and each syllable
// starts with its key (libime matches initials and partial finals itself; this only drops partial-sentence results).
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
#include <set>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> out; size_t a = 0, b;
    while ((b = s.find('\'', a)) != std::string::npos) { out.push_back(s.substr(a, b - a)); a = b + 1; }
    out.push_back(s.substr(a));
    return out;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s lm sc.dict sentences.tsv nbest\n", argv[0]); return 1; }
    libime::PinyinIME ime(std::make_unique<libime::PinyinDictionary>(), std::make_unique<libime::UserLanguageModel>(argv[1]));
    ime.dict()->load(libime::PinyinDictionary::SystemDict, argv[2], libime::PinyinDictFormat::Binary);
    ime.setNBest(std::stoul(argv[4])); ime.setBeamSize(20); ime.setFrameSize(10); ime.setPartialLongWordLimit(4);
    ime.setFuzzyFlags(libime::PinyinFuzzyFlags{libime::PinyinFuzzyFlag::VE_UE} | libime::PinyinFuzzyFlag::Inner |
                      libime::PinyinFuzzyFlag::InnerShort | libime::PinyinFuzzyFlag::PartialFinal);
    libime::PinyinContext c(&ime);
    std::ifstream f(argv[3]); std::string l;
    while (std::getline(f, l)) {
        auto t = l.find('\t'); if (t == std::string::npos) continue;
        std::string py = l.substr(0, t), ref = l.substr(t + 1), keys;
        auto t2 = ref.find('\t');
        if (t2 != std::string::npos) { keys = ref.substr(t2 + 1); ref = ref.substr(0, t2); }
        std::vector<std::string> kv = keys.empty() ? std::vector<std::string>{} : split(keys);
        c.clear();
        auto a = std::chrono::steady_clock::now();
        c.type(py);
        const auto &cs = c.candidates();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        printf("{\"py\":\"%s\",\"ref\":\"%s\",\"keys\":\"%s\",\"decode_ms\":%.3f,\"cands\":[", py.c_str(), ref.c_str(),
               keys.c_str(), ms);
        std::set<std::string> seen; bool first = true;
        for (size_t i = 0; i < cs.size(); i++) {
            // a full-sentence candidate spells exactly the typed pinyin (ignoring syllable separators)
            std::string full = c.candidateFullPinyin(i), text = cs[i].toString();
            bool ok;
            if (kv.empty()) {
                full.erase(std::remove(full.begin(), full.end(), '\''), full.end());
                ok = full == py;
            } else {
                auto sy = split(full);
                ok = sy.size() == kv.size();
                for (size_t j = 0; ok && j < sy.size(); j++) ok = sy[j].compare(0, kv[j].size(), kv[j]) == 0;
            }
            if (!ok || !seen.insert(text).second) continue;
            printf("%s{\"text\":\"%s\",\"score\":%.4f,\"rank\":%zu}", first ? "" : ",", text.c_str(), cs[i].score(), i);
            first = false;
        }
        printf("]}\n");
    }
    return 0;
}
