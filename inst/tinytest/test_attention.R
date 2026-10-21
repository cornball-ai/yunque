# Relative-position attention vs base-R references (needs anvl + PJRT CPU).

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

nva <- function(a) anvl::nv_array(c(a), dtype = "f32", shape = dim(a))
reltol <- function(got, ref) max(abs(got - ref)) / max(abs(ref))

# rel_shift: out[i, t] = x[i, T - i + t] (hand-verified from the pad/reshape/
# slice trick).
ref_rel_shift <- function(x) {
    d <- dim(x)
    B <- d[1]
    H <- d[2]
    T <- d[3]
    out <- array(0, c(B, H, T, T))
    for (b in 1:B) {
        for (h in 1:H) {
            for (i in 1:T) {
                for (t in 1:T) {
                    out[b, h, i, t] <- x[b, h, i, T - i + t]
                }
            }
        }
    }
    out
}

set.seed(3)
B <- 2L
H <- 2L
T <- 4L
dk <- 3L
P <- 2L * T - 1L

xrs <- array(rnorm(B * H * T * P), c(B, H, T, P))
expect_equal(dim(as.array(rel_shift(nva(xrs)))), c(B, H, T, T))
expect_true(reltol(as.array(rel_shift(nva(xrs))), ref_rel_shift(xrs)) < 1e-5)

# full relative-position attention
q <- array(rnorm(B * H * T * dk), c(B, H, T, dk))
k <- array(rnorm(B * H * T * dk), c(B, H, T, dk))
v <- array(rnorm(B * H * T * dk), c(B, H, T, dk))
p <- array(rnorm(B * H * P * dk), c(B, H, P, dk))
bu <- matrix(rnorm(H * dk), H, dk)
bv <- matrix(rnorm(H * dk), H, dk)

ref_rpa <- function(q, k, v, p, bu, bv) {
    d <- dim(q)
    B <- d[1]
    H <- d[2]
    T <- d[3]
    dk <- d[4]
    scale <- 1 / sqrt(dk)
    out <- array(0, c(B, H, T, dk))
    for (b in 1:B) {
        for (h in 1:H) {
            qh <- matrix(q[b, h, , ], T, dk)
            kh <- matrix(k[b, h, , ], T, dk)
            vh <- matrix(v[b, h, , ], T, dk)
            ph <- matrix(p[b, h, , ], dim(p)[3], dk)
            qu <- sweep(qh, 2, bu[h, ], "+")
            qv <- sweep(qh, 2, bv[h, ], "+")
            ac <- qu %*% t(kh)
            bd <- qv %*% t(ph)
            bds <- matrix(0, T, T)
            for (i in 1:T) {
                for (t in 1:T) {
                    bds[i, t] <- bd[i, T - i + t]
                }
            }
            sc <- (ac + bds) * scale
            at <- t(apply(sc, 1, function(r) exp(r - max(r)) / sum(exp(r - max(r)))))
            out[b, h, , ] <- at %*% vh
        }
    }
    out
}

got <- as.array(rel_position_attention(nva(q), nva(k), nva(v), nva(p),
                                       nva(bu), nva(bv)))
ref <- ref_rpa(q, k, v, p, bu, bv)
expect_equal(dim(got), c(B, H, T, dk))
expect_true(reltol(got, ref) < 1e-4)

# additive mask: a large negative bias zeroes out the last key everywhere
mask <- array(0, c(B, H, T, T))
mask[, , , T] <- -1e4
gotm <- as.array(rel_position_attention(nva(q), nva(k), nva(v), nva(p),
                                        nva(bu), nva(bv), mask = nva(mask)))
# the masked output should differ from the unmasked one
expect_true(max(abs(gotm - got)) > 1e-3)

# pos_emb with batch 1 broadcasts across the batch
p1 <- array(p[1, , , ], c(1L, H, P, dk))
got_b1 <- as.array(rel_position_attention(nva(q), nva(k), nva(v), nva(p1),
                                          nva(bu), nva(bv)))
# equals running with p[1] copied into every batch slot
p_rep <- p
for (b in 1:B) {
    p_rep[b, , , ] <- p[1, , , ]
}
expect_true(reltol(got_b1, ref_rpa(q, k, v, p_rep, bu, bv)) < 1e-4)

# repeat_kv: query head j reads KV head ceiling(j / groups), each KV head
# repeated `groups` times consecutively.
kvx <- array(rnorm(2 * 2 * 3 * 4), c(2L, 2L, 3L, 4L))
grp <- 3L
kref <- kvx[, rep(1:2, each = grp), , ]
kgot <- as.array(repeat_kv(nva(kvx), grp))
expect_equal(dim(kgot), c(2L, 6L, 3L, 4L))
expect_true(max(abs(kgot - kref)) < 1e-6)

# rope_apply: interleaved pairs (2k - 1, 2k) rotate by the angle in
# cos/sin, which carry each angle twice.
Dr <- 6L
Sr <- 3L
rx <- array(rnorm(B * H * Sr * Dr), c(B, H, Sr, Dr))
ang <- matrix(runif(Sr * Dr %/% 2L, 0, pi), Sr, Dr %/% 2L)
rc <- cos(ang)[, rep(seq_len(Dr %/% 2L), each = 2L)]
rs <- sin(ang)[, rep(seq_len(Dr %/% 2L), each = 2L)]
ref_rope <- function(x, cs, sn) {
    out <- x
    for (b in 1:B) {
        for (h in 1:H) {
            for (s in 1:Sr) {
                for (k in seq_len(Dr %/% 2L)) {
                    re <- x[b, h, s, 2L * k - 1L]
                    im <- x[b, h, s, 2L * k]
                    out[b, h, s, 2L * k - 1L] <- re * cs[b, s, 2L * k - 1L] -
                        im * sn[b, s, 2L * k - 1L]
                    out[b, h, s, 2L * k] <- im * cs[b, s, 2L * k] +
                        re * sn[b, s, 2L * k]
                }
            }
        }
    }
    out
}
rc3 <- aperm(array(rc, c(Sr, Dr, B)), c(3L, 1L, 2L))
rs3 <- aperm(array(rs, c(Sr, Dr, B)), c(3L, 1L, 2L))
rc4 <- aperm(array(rc3, c(B, Sr, Dr, H)), c(1L, 4L, 2L, 3L))
rs4 <- aperm(array(rs3, c(B, Sr, Dr, H)), c(1L, 4L, 2L, 3L))
ra <- as.array(rope_apply(nva(rx), nva(rc4), nva(rs4)))
expect_true(max(abs(ra - ref_rope(rx, rc3, rs3))) < 1e-5)

# rope_split: element i pairs with i + D/2
hc <- cos(ang)
hs <- sin(ang)
r2 <- Dr %/% 2L
sref <- rx
for (b in 1:B) {
    for (h in 1:H) {
        x1 <- rx[b, h, , 1:r2]
        x2 <- rx[b, h, , (r2 + 1L):Dr]
        sref[b, h, , 1:r2] <- x1 * hc - x2 * hs
        sref[b, h, , (r2 + 1L):Dr] <- x2 * hc + x1 * hs
    }
}
hc4 <- aperm(array(hc, c(Sr, r2, B, H)), c(3L, 4L, 1L, 2L))
hs4 <- aperm(array(hs, c(Sr, r2, B, H)), c(3L, 4L, 1L, 2L))
sgot <- as.array(rope_split(nva(rx), nva(hc4), nva(hs4)))
expect_true(max(abs(sgot - sref)) < 1e-5)

# Lower-rank RoPE tables: [S, D] is shared across batch and heads, and a
# per-batch [B, S, D] table gets the head axis inserted.
ra2 <- as.array(rope_apply(nva(rx), nva(rc), nva(rs)))
expect_true(max(abs(ra2 - ref_rope(rx, rc3, rs3))) < 1e-5)
sgot2 <- as.array(rope_split(nva(rx), nva(hc), nva(hs)))
expect_true(max(abs(sgot2 - sref)) < 1e-5)
rc3[2, , ] <- rc3[2, , ] * 0.5
rs3[2, , ] <- rs3[2, , ] * 0.5
ra3 <- as.array(rope_apply(nva(rx), nva(rc3), nva(rs3)))
expect_true(max(abs(ra3 - ref_rope(rx, rc3, rs3))) < 1e-5)
