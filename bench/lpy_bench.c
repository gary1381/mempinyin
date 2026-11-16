// Sentence accuracy + per-keystroke latency for libpinyin (engine behind ibus-libpinyin).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
typedef struct _ctx ctx_t; typedef struct _inst inst_t;
ctx_t *pinyin_init(const char *systemdir, const char *userdir);
bool pinyin_set_options(ctx_t *, unsigned int);
inst_t *pinyin_alloc_instance(ctx_t *);
size_t pinyin_parse_more_full_pinyins(inst_t *, const char *);
bool pinyin_guess_sentence(inst_t *);
bool pinyin_get_sentence(inst_t *, unsigned char index, char **sentence);
bool pinyin_reset(inst_t *);
void g_free(void *);
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static long rss_kb(void) { FILE *f = fopen("/proc/self/status", "r"); char l[256]; long v = -1;
  while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS:", 6)) v = atol(l + 6); if (f) fclose(f); return v; }
static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
  long r0 = rss_kb(); double t0 = now_us();
  ctx_t *c = pinyin_init(argv[1], argv[2]);
  if (!c) { fprintf(stderr, "init failed\n"); return 1; }
  pinyin_set_options(c, (1u << 1) | (1u << 3) | (1u << 7)); // IS_PINYIN | PINYIN_INCOMPLETE | DYNAMIC_ADJUST
  inst_t *in = pinyin_alloc_instance(c);
  printf("open_ms=%.2f rss_before_kb=%ld rss_after_kb=%ld\n", (now_us() - t0) / 1e3, r0, rss_kb());
  FILE *f = fopen(argv[3], "r"); char py[256][128], hz[256][256], l[512]; int n = 0, ok = 0;
  while (fgets(l, sizeof l, f)) { char *t = strchr(l, '\t'); *t = 0; t[1 + strcspn(t + 1, "\r\n")] = 0; strcpy(py[n], l); strcpy(hz[n], t + 1); n++; }
  for (int i = 0; i < n; i++) { pinyin_reset(in); pinyin_parse_more_full_pinyins(in, py[i]); pinyin_guess_sentence(in);
    char *s = NULL; pinyin_get_sentence(in, 0, &s); int good = s && !strcmp(s, hz[i]); ok += good;
    printf("S\t%s\t%s\t%s\t%s\n", good ? "OK" : "XX", py[i], hz[i], s ? s : "(null)"); g_free(s); }
  printf("sent_n=%d top1_exact=%.1f%%\n", n, 100.0 * ok / n);
  static double ks[100000]; int nk = 0;
  for (int it = 0; it < 5; it++) for (int i = 0; i < n; i++) { int L = strlen(py[i]); char buf[128];
    for (int k = 1; k <= L; k++) { memcpy(buf, py[i], k); buf[k] = 0; double a = now_us();
      pinyin_reset(in); pinyin_parse_more_full_pinyins(in, buf); pinyin_guess_sentence(in);
      char *s = NULL; pinyin_get_sentence(in, 0, &s); g_free(s); ks[nk++] = now_us() - a; } }
  qsort(ks, nk, sizeof(double), cmpd);
  printf("keystroke_n=%d p50_us=%.1f p95_us=%.1f p99_us=%.1f max_us=%.1f rss_end_kb=%ld\n", nk, ks[nk/2], ks[nk*95/100], ks[nk*99/100], ks[nk-1], rss_kb());
  return 0;
}
