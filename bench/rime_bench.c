// Benchmark for librime (luna_pinyin_simp): accuracy, selection actions, per-keystroke latency.
#include <rime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static RimeApi *api;
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static long rss_kb(void) { FILE *f = fopen("/proc/self/status", "r"); char l[256]; long v = -1;
  while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS:", 6)) v = atol(l + 6); if (f) fclose(f); return v; }
static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
typedef struct { char py[128]; char hz[256]; } Item;
static int load(const char *fn, Item *it, int max) { FILE *f = fopen(fn, "r"); char l[512]; int n = 0;
  while (f && n < max && fgets(l, sizeof l, f)) { char *t = strchr(l, '\t'); if (!t) continue; *t = 0;
    t[1 + strcspn(t + 1, "\r\n")] = 0; strcpy(it[n].py, l); strcpy(it[n].hz, t + 1); n++; }
  if (f) fclose(f); return n; }

// Rank of target among the first `max` candidates (-1 if absent); copies candidate 0 into top.
static int rank_of(RimeSessionId s, const char *target, int max, char *top) {
  RimeCandidateListIterator it = {0}; int r = -1, k = 0; top[0] = 0;
  if (!api->candidate_list_begin(s, &it)) return -1;
  while (k < max && api->candidate_list_next(&it)) {
    if (k == 0) strcpy(top, it.candidate.text);
    if (r < 0 && !strcmp(it.candidate.text, target)) r = k;
    k++;
  }
  api->candidate_list_end(&it); return r;
}

int main(int argc, char **argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s shared_dir user_dir schema sentences.tsv words.tsv\n", argv[0]); return 1; }
  RIME_STRUCT(RimeTraits, traits);
  traits.shared_data_dir = argv[1]; traits.user_data_dir = argv[2];
  traits.distribution_name = "bench"; traits.distribution_code_name = "bench"; traits.distribution_version = "1";
  traits.app_name = "rime.bench"; traits.min_log_level = 2;
  api = rime_get_api();
  long r0 = rss_kb(); double t0 = now_us();
  api->setup(&traits); api->initialize(&traits);
  if (api->start_maintenance(True)) api->join_maintenance_thread();
  double tdeploy = now_us() - t0; t0 = now_us();
  RimeSessionId s = api->create_session();
  if (!api->select_schema(s, argv[3])) { fprintf(stderr, "select_schema failed\n"); return 1; }
  api->simulate_key_sequence(s, "a"); api->clear_composition(s);  // force dictionary load
  printf("deploy_ms=%.1f open_ms=%.2f rss_before_kb=%ld rss_after_kb=%ld\n", tdeploy / 1e3, (now_us() - t0) / 1e3, r0, rss_kb());

  static Item S[256], W[128]; int ns = load(argv[4], S, 256), nw = load(argv[5], W, 128);
  char top[512]; int exact = 0, in3 = 0, in10 = 0, steps_total = 0, fail = 0;
  for (int i = 0; i < ns; i++) {
    api->clear_composition(s); api->simulate_key_sequence(s, S[i].py);
    int r = rank_of(s, S[i].hz, 10, top);
    exact += r == 0; in3 += r >= 0 && r < 3; in10 += r >= 0;
    printf("S\t%s\t%s\t%s\t%s\n", r == 0 ? "OK" : "XX", S[i].py, S[i].hz, top);
    // greedy selection; Rime commits once every segment is selected
    char done_txt[512] = ""; int steps = 0, done = 0; size_t fixed = 0;
    while (steps < 12) {
      const char *rest = S[i].hz + fixed; RimeCandidateListIterator it = {0}; int found = -1, k = 0; size_t best = 0;
      if (api->candidate_list_begin(s, &it)) {
        while (k < 60 && api->candidate_list_next(&it)) { size_t l = strlen(it.candidate.text);
          if (!strcmp(it.candidate.text, rest)) { found = k; best = l; break; }
          if (l > best && !strncmp(rest, it.candidate.text, l)) { best = l; found = k; } k++; }
        api->candidate_list_end(&it); }
      if (found < 0) break;
      api->select_candidate(s, found); steps++; fixed += best;
      RIME_STRUCT(RimeCommit, cm);
      if (api->get_commit(s, &cm)) { strcpy(done_txt, cm.text); api->free_commit(&cm); done = !strcmp(done_txt, S[i].hz); break; }
    }
    if (!done) fail++; else steps_total += steps;  // in Rime the final select is the commit
  }
  printf("sent_n=%d top1_exact=%.1f%% in_top3=%.1f%% in_top10=%.1f%% avg_actions=%.2f unreachable=%d\n", ns,
         100.0 * exact / ns, 100.0 * in3 / ns, 100.0 * in10 / ns, (double)steps_total / (ns - fail), fail);

  int w1 = 0, w5 = 0;
  for (int i = 0; i < nw; i++) { api->clear_composition(s); api->simulate_key_sequence(s, W[i].py);
    int r = rank_of(s, W[i].hz, 300, top); w1 += r == 0; w5 += r >= 0 && r < 5;
    printf("W\t%s\t%s\trank=%d\ttop1=%s\n", W[i].py, W[i].hz, r, top); }
  printf("word_n=%d top1=%.1f%% top5=%.1f%%\n", nw, 100.0 * w1 / nw, 100.0 * w5 / nw);

  static double ks[200000], full[20000]; int nk = 0, nf = 0; char key[2] = {0};
  for (int it = 0; it < 30; it++) for (int i = 0; i < ns; i++) {
    api->clear_composition(s);
    for (const char *p = S[i].py; *p; p++) { key[0] = *p; double a = now_us();
      api->simulate_key_sequence(s, key); RIME_STRUCT(RimeContext, ctx); api->get_context(s, &ctx); api->free_context(&ctx);
      ks[nk++] = now_us() - a; }
    api->clear_composition(s); double a = now_us(); api->simulate_key_sequence(s, S[i].py);
    RIME_STRUCT(RimeContext, ctx); api->get_context(s, &ctx); api->free_context(&ctx); full[nf++] = now_us() - a;
  }
  qsort(ks, nk, sizeof(double), cmpd); qsort(full, nf, sizeof(double), cmpd);
  printf("keystroke_n=%d p50_us=%.1f p95_us=%.1f p99_us=%.1f max_us=%.1f\n", nk, ks[nk/2], ks[nk*95/100], ks[nk*99/100], ks[nk-1]);
  printf("fullsent_n=%d p50_us=%.1f p95_us=%.1f p99_us=%.1f max_us=%.1f\n", nf, full[nf/2], full[nf*95/100], full[nf*99/100], full[nf-1]);
  printf("rss_end_kb=%ld\n", rss_kb());
  api->destroy_session(s); api->finalize(); return 0;
}
