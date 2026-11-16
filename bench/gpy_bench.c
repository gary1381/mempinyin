// Benchmark for libgooglepinyin (AOSP PinyinIME core) using its C API.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>

typedef unsigned short char16;
bool im_open_decoder(const char *fn_sys_dict, const char *fn_usr_dict);
void im_close_decoder(void);
void im_set_max_lens(size_t max_sps_len, size_t max_hzs_len);
void im_reset_search(void);
size_t im_search(const char *sps_buf, size_t sps_len);
char16 *im_get_candidate(size_t cand_id, char16 *cand_str, size_t max_len);
size_t im_choose(size_t cand_id);
size_t im_get_fixed_len(void);
void im_flush_cache(void);

#define MAXC 128
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static long rss_kb(void) {
  FILE *f = fopen("/proc/self/status", "r"); char l[256]; long v = -1;
  while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS:", 6)) v = atol(l + 6);
  if (f) fclose(f); return v;
}
// UTF-8 <-> UTF-16 (BMP only, enough for CJK)
static int u8to16(const char *s, char16 *o) { int n = 0; const unsigned char *p = (const unsigned char *)s;
  while (*p) { if (*p < 0x80) o[n++] = *p++; else if ((*p & 0xE0) == 0xC0) { o[n++] = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F); p += 2; }
    else { o[n++] = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); p += 3; } }
  o[n] = 0; return n; }
static void u16to8(const char16 *s, char *o) { for (; *s; s++) { unsigned c = *s;
  if (c < 0x80) *o++ = c; else if (c < 0x800) { *o++ = 0xC0 | (c >> 6); *o++ = 0x80 | (c & 0x3F); }
  else { *o++ = 0xE0 | (c >> 12); *o++ = 0x80 | ((c >> 6) & 0x3F); *o++ = 0x80 | (c & 0x3F); } } *o = 0; }
static int u16len(const char16 *s) { int n = 0; while (s[n]) n++; return n; }
static void norm_ta(char16 *s) { for (; *s; s++) if (*s == 0x5979 || *s == 0x5B83) *s = 0x4ED6; } // 她/它 -> 他
static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
static int edit(const char16 *a, int n, const char16 *b, int m) {
  int d[MAXC + 1][MAXC + 1];
  for (int i = 0; i <= n; i++) d[i][0] = i; for (int j = 0; j <= m; j++) d[0][j] = j;
  for (int i = 1; i <= n; i++) for (int j = 1; j <= m; j++) { int c = d[i-1][j-1] + (a[i-1] != b[j-1]);
    if (d[i-1][j] + 1 < c) c = d[i-1][j] + 1; if (d[i][j-1] + 1 < c) c = d[i][j-1] + 1; d[i][j] = c; }
  return d[n][m]; }

typedef struct { char py[128]; char hz[256]; } Item;
static int load(const char *fn, Item *it, int max) { FILE *f = fopen(fn, "r"); char l[512]; int n = 0;
  while (f && n < max && fgets(l, sizeof l, f)) { char *t = strchr(l, '\t'); if (!t) continue; *t = 0;
    char *e = t + 1 + strcspn(t + 1, "\r\n"); *e = 0; strcpy(it[n].py, l); strcpy(it[n].hz, t + 1); n++; }
  if (f) fclose(f); return n; }

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s sysdict usrdict sentences.tsv words.tsv\n", argv[0]); return 1; }
  long r0 = rss_kb(); double t0 = now_us();
  if (!im_open_decoder(argv[1], argv[2])) { fprintf(stderr, "open failed\n"); return 1; }
  double topen = now_us() - t0; long r1 = rss_kb();
  im_set_max_lens(64, 32);
  printf("open_ms=%.2f rss_before_kb=%ld rss_after_kb=%ld\n", topen / 1e3, r0, r1);

  static Item S[256], W[128]; int ns = load(argv[3], S, 256), nw = load(argv[4], W, 128);
  char16 cand[MAXC], tgt[MAXC]; char out[512];

  // 1) accuracy: top-1 whole-sentence, char error rate, selection steps (greedy oracle)
  int exact = 0, exact_ta = 0, errs = 0, chars = 0, steps_total = 0, steps_fail = 0;
  for (int i = 0; i < ns; i++) {
    int tl = u8to16(S[i].hz, tgt); chars += tl;
    im_reset_search(); size_t nc = im_search(S[i].py, strlen(S[i].py));
    im_get_candidate(0, cand, MAXC); u16to8(cand, out);
    int cl = u16len(cand); int e = edit(cand, cl, tgt, tl); errs += e;
    char16 a[MAXC], b[MAXC]; memcpy(a, cand, sizeof cand); memcpy(b, tgt, sizeof tgt); norm_ta(a); norm_ta(b);
    int ok = (e == 0), okta = (edit(a, u16len(a), b, u16len(b)) == 0); exact += ok; exact_ta += okta;
    // greedy selection: each choose of a non-whole candidate is one extra action
    int steps = 0, done = ok;
    while (!done && steps < 12) {
      size_t fixed = im_get_fixed_len(); int found = -1, best = 0;
      for (size_t c = 1; c < nc && c < 60; c++) { im_get_candidate(c, cand, MAXC); int l = u16len(cand);
        if (l > best && fixed + l <= (size_t)tl && !memcmp(cand, tgt + fixed, l * 2)) { best = l; found = c; } }
      if (found < 0) break;
      nc = im_choose(found); steps++;
      im_get_candidate(0, cand, MAXC);
      if (im_get_fixed_len() >= (size_t)tl || (u16len(cand) == tl && !memcmp(cand, tgt, tl * 2))) done = 1;
    }
    if (!done) steps_fail++; else steps_total += steps + 1; // +1 = final commit
    printf("S\t%s\t%s\t%s\t%s\n", ok ? "OK" : (okta ? "OK~" : "XX"), S[i].py, S[i].hz, out);
  }
  printf("sent_n=%d top1_exact=%.1f%% top1_exact_ta_norm=%.1f%% CER=%.2f%% avg_actions=%.2f unreachable=%d\n",
         ns, 100.0 * exact / ns, 100.0 * exact_ta / ns, 100.0 * errs / chars,
         (double)steps_total / (ns - steps_fail), steps_fail);

  // 2) word / abbreviation recall
  int w1 = 0, w5 = 0;
  for (int i = 0; i < nw; i++) { u8to16(W[i].hz, tgt); im_reset_search(); size_t nc = im_search(W[i].py, strlen(W[i].py));
    int rank = -1; char first[256] = "";
    for (size_t c = 0; c < nc && c < 10; c++) { im_get_candidate(c, cand, MAXC); if (c == 0) u16to8(cand, first);
      if (!memcmp(cand, tgt, (u16len(tgt) + 1) * 2)) { rank = c; break; } }
    w1 += rank == 0; w5 += rank >= 0 && rank < 5;
    printf("W\t%s\t%s\trank=%d\ttop1=%s\n", W[i].py, W[i].hz, rank, first); }
  printf("word_n=%d top1=%.1f%% top5=%.1f%%\n", nw, 100.0 * w1 / nw, 100.0 * w5 / nw);

  // 3) latency: per-keystroke incremental search + fetch 10 candidates (what a UI does per key)
  static double ks[200000]; int nk = 0; static double full[20000]; int nf = 0; int iters = 30;
  for (int it = 0; it < iters; it++) for (int i = 0; i < ns; i++) { int L = strlen(S[i].py);
    im_reset_search();
    for (int k = 1; k <= L; k++) { double a = now_us(); size_t nc = im_search(S[i].py, k);
      for (size_t c = 0; c < nc && c < 10; c++) im_get_candidate(c, cand, MAXC); ks[nk++] = now_us() - a; }
    im_reset_search(); double a = now_us(); size_t nc = im_search(S[i].py, L);
    for (size_t c = 0; c < nc && c < 10; c++) im_get_candidate(c, cand, MAXC); full[nf++] = now_us() - a; }
  qsort(ks, nk, sizeof(double), cmpd); qsort(full, nf, sizeof(double), cmpd);
  printf("keystroke_n=%d p50_us=%.1f p95_us=%.1f p99_us=%.1f max_us=%.1f\n", nk, ks[nk/2], ks[nk*95/100], ks[nk*99/100], ks[nk-1]);
  printf("fullsent_n=%d p50_us=%.1f p95_us=%.1f p99_us=%.1f max_us=%.1f\n", nf, full[nf/2], full[nf*95/100], full[nf*99/100], full[nf-1]);
  printf("rss_end_kb=%ld\n", rss_kb());
  im_close_decoder(); return 0;
}
