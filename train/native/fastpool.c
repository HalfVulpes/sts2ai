// Native hot path of the rollout collector (loaded with ctypes, see sts2ai/fastpool.py).
//
// The simulator workers are separate processes. Each one writes an observation into its own
// shared-memory slot, then writes one byte "R" to its stdout pipe and blocks reading a 16-byte
// command from stdin. This library owns those pipes:
//   pool_wait    epoll over all worker stdouts; returns a batch of ready workers
//   pool_heads   copy their slot headers
//   pool_gather  copy their observations into dense batch arrays padded to the batch maximum
//   pool_send    write commands (STEP action / RESET) to many workers
// plus gae(), the per-environment reverse scan for GAE / lambda-returns.
// Everything here runs without the Python GIL (ctypes releases it).
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define HEADER_INTS 32
#define EINTR_CODE (-2000000) // interrupted by a signal: return to Python so its handler can run
#define H_NTOK 2
#define H_NACT 3

typedef struct {
    int n;
    int epfd;
    int *in_fd, *out_fd;
    uint8_t **slot;
    long slot_bytes;
    long off_tokcat, off_toknum, off_actcat, off_actref, off_actnum;
    int n_tok_cat, f_tok, n_act_cat, f_act;
    struct epoll_event *events;
    uint8_t *ready;   // ready[i] = observation available and not yet returned
} Pool;

static long now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000L + t.tv_nsec;
}

// layout = {slot_bytes, off_tokcat, off_toknum, off_actcat, off_actref, off_actnum,
//           n_tok_cat, f_tok, n_act_cat, f_act}
Pool *pool_create(int n, const char **paths, const int *in_fds, const int *out_fds, const long *layout) {
    Pool *p = calloc(1, sizeof(Pool));
    p->n = n;
    p->slot_bytes = layout[0];
    p->off_tokcat = layout[1]; p->off_toknum = layout[2]; p->off_actcat = layout[3];
    p->off_actref = layout[4]; p->off_actnum = layout[5];
    p->n_tok_cat = (int)layout[6]; p->f_tok = (int)layout[7]; p->n_act_cat = (int)layout[8]; p->f_act = (int)layout[9];
    p->in_fd = malloc(sizeof(int) * n);
    p->out_fd = malloc(sizeof(int) * n);
    p->slot = calloc(n, sizeof(uint8_t *));
    p->events = malloc(sizeof(struct epoll_event) * n);
    p->ready = calloc(n, 1);
    p->epfd = epoll_create1(0);
    if (p->epfd < 0) return NULL;
    for (int i = 0; i < n; i++) {
        p->in_fd[i] = in_fds[i];
        p->out_fd[i] = out_fds[i];
        int fd = open(paths[i], O_RDWR);
        if (fd < 0) return NULL;
        void *m = mmap(NULL, p->slot_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (m == MAP_FAILED) return NULL;
        p->slot[i] = m;
        fcntl(out_fds[i], F_SETFL, fcntl(out_fds[i], F_GETFL) | O_NONBLOCK);
        struct epoll_event ev = {.events = EPOLLIN, .data.u32 = (uint32_t)i};
        if (epoll_ctl(p->epfd, EPOLL_CTL_ADD, out_fds[i], &ev) < 0) return NULL;
    }
    return p;
}

void pool_destroy(Pool *p) {
    if (!p) return;
    for (int i = 0; i < p->n; i++)
        if (p->slot[i]) munmap(p->slot[i], p->slot_bytes);
    close(p->epfd);
    free(p->in_fd); free(p->out_fd); free(p->slot); free(p->events); free(p->ready);
    free(p);
}

// Drain readable pipes once. Returns number of newly ready workers, or -(i+1) if worker i died.
static int poll_once(Pool *p, int timeout_ms_or_neg, long timeout_ns) {
    int nev;
    if (timeout_ms_or_neg >= 0) nev = epoll_wait(p->epfd, p->events, p->n, timeout_ms_or_neg);
    else {
        struct timespec ts = {.tv_sec = timeout_ns / 1000000000L, .tv_nsec = timeout_ns % 1000000000L};
        nev = epoll_pwait2(p->epfd, p->events, p->n, &ts, NULL);
    }
    if (nev < 0) return errno == EINTR ? EINTR_CODE : -1000000;
    int got = 0;
    char buf[64];
    for (int e = 0; e < nev; e++) {
        int i = (int)p->events[e].data.u32;
        ssize_t r = read(p->out_fd[i], buf, sizeof buf);
        if (r == 0) return -(i + 1);
        if (r < 0) {
            if (errno == EAGAIN || errno == EINTR) continue;
            return -(i + 1);
        }
        if (!p->ready[i]) { p->ready[i] = 1; got++; }
    }
    return got;
}

// Wait for ready workers. Blocks up to first_timeout_ms for the first one; after the first arrives,
// keeps collecting until min_ready are ready or batch_wait_us has passed. Writes ready ids (ascending)
// to out_ids and clears them. Returns count (0 on timeout), or a negative error (-(i+1): worker i died).
int pool_wait(Pool *p, int min_ready, int first_timeout_ms, long batch_wait_us, int *out_ids) {
    int count = 0;
    for (int i = 0; i < p->n; i++) count += p->ready[i];
    long t_first = count > 0 ? now_ns() : 0;
    long t0 = now_ns();
    while (count < min_ready) {
        int r;
        if (count == 0) {
            long left_ms = first_timeout_ms - (now_ns() - t0) / 1000000L;
            if (left_ms <= 0) break;
            r = poll_once(p, (int)left_ms, 0);
            if (r > 0) t_first = now_ns();
        } else {
            long left = batch_wait_us * 1000L - (now_ns() - t_first);
            if (left <= 0) break;
            r = poll_once(p, -1, left);
        }
        if (r == EINTR_CODE) break;
        if (r < 0) return r;
        count += r;
    }
    if (count > 0 && count < p->n) {
        int r = poll_once(p, 0, 0); // anything else already readable
        if (r < 0 && r != EINTR_CODE) return r;
        if (r > 0) count += r;
    }
    int k = 0;
    for (int i = 0; i < p->n; i++)
        if (p->ready[i]) { out_ids[k++] = i; p->ready[i] = 0; }
    return k;
}

void pool_heads(Pool *p, const int *ids, int n, int32_t *out) {
    for (int k = 0; k < n; k++) memcpy(out + (long)k * HEADER_INTS, p->slot[ids[k]], HEADER_INTS * 4);
}

// Dense batch, padded to T tokens / A actions (callers pass the batch maxima).
// tc int16[n,T,n_tok_cat]  tn float32[n,T,f_tok]  ac int16[n,A,n_act_cat]  ar int16[n,A,2]  an float32[n,A,f_act]
void pool_gather(Pool *p, const int *ids, int n, int T, int A,
                 int16_t *tc, float *tn, int16_t *ac, int16_t *ar, float *an) {
    for (int k = 0; k < n; k++) {
        const uint8_t *s = p->slot[ids[k]];
        const int32_t *h = (const int32_t *)s;
        int nt = h[H_NTOK], na = h[H_NACT];
        if (nt > T) nt = T;
        if (na > A) na = A;
        int16_t *tck = tc + (long)k * T * p->n_tok_cat;
        float *tnk = tn + (long)k * T * p->f_tok;
        int16_t *ack = ac + (long)k * A * p->n_act_cat;
        int16_t *ark = ar + (long)k * A * 2;
        float *ank = an + (long)k * A * p->f_act;
        memcpy(tck, s + p->off_tokcat, (size_t)nt * p->n_tok_cat * 2);
        memset(tck + (long)nt * p->n_tok_cat, 0, (size_t)(T - nt) * p->n_tok_cat * 2);
        memcpy(tnk, s + p->off_toknum, (size_t)nt * p->f_tok * 4);
        memset(tnk + (long)nt * p->f_tok, 0, (size_t)(T - nt) * p->f_tok * 4);
        memcpy(ack, s + p->off_actcat, (size_t)na * p->n_act_cat * 2);
        memset(ack + (long)na * p->n_act_cat, 0, (size_t)(A - na) * p->n_act_cat * 2);
        memcpy(ark, s + p->off_actref, (size_t)na * 2 * 2);
        for (long j = (long)na * 2; j < (long)A * 2; j++) ark[j] = -1;
        memcpy(ank, s + p->off_actnum, (size_t)na * p->f_act * 4);
        memset(ank + (long)na * p->f_act, 0, (size_t)(A - na) * p->f_act * 4);
    }
}

// Write a 16-byte command (int32 cmd, int32 a, int64 b) to each worker. a/b may be NULL (zeros).
// Returns 0, or -(i+1) if writing to worker i failed.
int pool_send(Pool *p, const int *ids, int n, int cmd, const int32_t *a, const int64_t *b) {
    uint8_t frame[16];
    for (int k = 0; k < n; k++) {
        int32_t av = a ? a[k] : 0;
        int64_t bv = b ? b[k] : 0;
        memcpy(frame, &cmd, 4);
        memcpy(frame + 4, &av, 4);
        memcpy(frame + 8, &bv, 8);
        int fd = p->in_fd[ids[k]];
        size_t off = 0;
        while (off < 16) {
            ssize_t w = write(fd, frame + off, 16 - off);
            if (w < 0) {
                if (errno == EINTR) continue;
                return -(ids[k] + 1);
            }
            off += (size_t)w;
        }
    }
    return 0;
}

// GAE over a rollout stored in collection order (rows of one env are in time order, envs interleaved).
//   done[i]: episode ended after step i (no bootstrap).  trunc[i]: cut by a simulator error
//   (bootstrap from the step's own value and don't carry advantage across).  next_value[i]: bootstrap
//   for the env's last step in this rollout (NaN -> own value).
// Writes advantages and lambda-returns (adv + value).
void gae(long n, const float *reward, const float *value, const uint8_t *done, const uint8_t *trunc,
         const float *next_value, const int32_t *env, int n_env, float gamma, float lam,
         float *adv, float *ret) {
    uint8_t *has_next = calloc(n_env, 1);
    float *nv = calloc(n_env, sizeof(float));
    float *carry = calloc(n_env, sizeof(float));
    for (long i = n - 1; i >= 0; i--) {
        int e = env[i];
        float a;
        if (trunc[i]) {
            a = reward[i] + (gamma - 1.0f) * value[i];
        } else if (done[i]) {
            a = reward[i] - value[i];
        } else {
            float next;
            if (has_next[e]) next = nv[e];
            else next = isnan(next_value[i]) ? value[i] : next_value[i];
            a = reward[i] + gamma * next - value[i];
            if (has_next[e]) a += gamma * lam * carry[e];
        }
        adv[i] = a;
        ret[i] = a + value[i];
        carry[e] = a;
        nv[e] = value[i];
        has_next[e] = 1;
    }
    free(has_next); free(nv); free(carry);
}
