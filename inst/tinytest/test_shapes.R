# Shape bucketing and attention masks (needs anvl + PJRT CPU).

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

nva <- function(a) anvl::nv_array(c(a), dtype = "f32", shape = dim(a))

set.seed(17)
x_r <- array(rnorm(2 * 5 * 3), c(2, 5, 3))
x <- nva(x_r)

# pad_to: 5 -> 8 along the sequence axis, zeros in the new positions
p <- pad_to(x, 4L)
expect_equal(anvl::shape(p), c(2L, 8L, 3L))
pa <- as.array(p)
expect_true(max(abs(pa[, 1:5, ] - x_r)) < 1e-7)
expect_true(all(pa[, 6:8, ] == 0))

# already a multiple: unchanged shape and values
same <- pad_to(x, 5L)
expect_equal(anvl::shape(same), c(2L, 5L, 3L))
expect_true(max(abs(as.array(same) - x_r)) < 1e-7)

# another axis, another fill value
p3 <- as.array(pad_to(x, 4L, dim = 3L, value = -1))
expect_equal(dim(p3), c(2L, 5L, 4L))
expect_true(all(p3[, , 4] == -1))
expect_true(max(abs(p3[, , 1:3] - x_r)) < 1e-7)

# seq_mask: 0 on valid keys, the bias on padded ones
m <- as.array(seq_mask(c(3L, 5L), 5L))
expect_equal(dim(m), c(2L, 1L, 1L, 5L))
expect_equal(m[1, 1, 1, ], c(0, 0, 0, -1e9, -1e9))
expect_equal(m[2, 1, 1, ], rep(0, 5))

# causal_mask: bias strictly above the diagonal
cm <- as.array(causal_mask(3L))
expect_equal(dim(cm), c(1L, 1L, 3L, 3L))
expect_equal(cm[1, 1, , ], matrix(c(0, 0, 0, -1e9, 0, 0, -1e9, -1e9, 0), 3, 3))

# a seq_mask fed to sdpa gives the same result as dropping the padded keys
B <- 2L
H <- 1L
S <- 5L
Dh <- 4L
q_r <- array(rnorm(B * H * S * Dh), c(B, H, S, Dh))
k_r <- array(rnorm(B * H * S * Dh), c(B, H, S, Dh))
v_r <- array(rnorm(B * H * S * Dh), c(B, H, S, Dh))
full <- as.array(sdpa(nva(q_r), nva(k_r), nva(v_r),
                      mask = seq_mask(c(3L, 5L), S)))
short <- as.array(sdpa(nva(q_r[1, , , , drop = FALSE]),
                       nva(k_r[1, , 1:3, , drop = FALSE]),
                       nva(v_r[1, , 1:3, , drop = FALSE])))
expect_true(max(abs(full[1, , , ] - short[1, , , ])) < 1e-5)
# batch 2 has no padding, so the mask leaves it alone
plain <- as.array(sdpa(nva(q_r), nva(k_r), nva(v_r)))
expect_true(max(abs(full[2, , , ] - plain[2, , , ])) < 1e-6)

# causal + padding masks combine with badd
both <- badd(causal_mask(S), seq_mask(c(3L, 5L), S))
expect_equal(anvl::shape(both), c(2L, 1L, 5L, 5L))
