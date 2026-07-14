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
