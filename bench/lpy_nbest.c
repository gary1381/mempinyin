// Oracle check: rank of the reference sentence in libpinyin's candidate list (N-best sentences come first).
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
typedef struct _ctx ctx_t; typedef struct _inst inst_t; typedef struct _cand cand_t;
ctx_t *pinyin_init(const char *, const char *); bool pinyin_set_options(ctx_t *, unsigned int);
inst_t *pinyin_alloc_instance(ctx_t *); size_t pinyin_parse_more_full_pinyins(inst_t *, const char *);
bool pinyin_guess_sentence(inst_t *); bool pinyin_guess_candidates(inst_t *, size_t, int);
bool pinyin_get_n_candidate(inst_t *, unsigned int *); bool pinyin_get_candidate(inst_t *, unsigned int, cand_t **);
bool pinyin_get_candidate_string(inst_t *, cand_t *, const char **); bool pinyin_reset(inst_t *);
int main(int argc, char **argv) {
  ctx_t *c = pinyin_init(argv[1], argv[2]); pinyin_set_options(c, (1u << 1) | (1u << 3) | (1u << 7));
  inst_t *in = pinyin_alloc_instance(c); FILE *f = fopen(argv[3], "r"); char l[512]; int n = 0, hit[10] = {0};
  while (fgets(l, sizeof l, f)) { char *t = strchr(l, '\t'); *t = 0; char *hz = t + 1; hz[strcspn(hz, "\r\n")] = 0; n++;
    pinyin_reset(in); pinyin_parse_more_full_pinyins(in, l); pinyin_guess_sentence(in); pinyin_guess_candidates(in, 0, 0);
    unsigned int nc = 0; pinyin_get_n_candidate(in, &nc); int r = -1;
    for (unsigned k = 0; k < nc && k < 10; k++) { cand_t *cd; const char *s = NULL; pinyin_get_candidate(in, k, &cd);
      pinyin_get_candidate_string(in, cd, &s); if (s && !strcmp(s, hz)) { r = k; break; } }
    if (r >= 0) for (int j = r; j < 10; j++) hit[j]++; }
  printf("n=%d", n); int ks[] = {0, 2, 4, 9}; for (int j = 0; j < 4; j++) printf(" in_top%d=%.1f%%", ks[j] + 1, 100.0 * hit[ks[j]] / n); printf("\n");
}
