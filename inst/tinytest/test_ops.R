# Unit checks of the composed ops against base-R reference math.
# Need anvl + a working PJRT CPU plugin; skipped elsewhere.

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

set.seed(7)
x_r <- matrix(rnorm(6 * 8), 6, 8)
x <- anvl::nv_array(x_r, dtype = "f32")

# softmax over last dim
sm <- as.array(yq_softmax(x))
ref <- t(apply(x_r, 1, function(r) exp(r - max(r)) / sum(exp(r - max(r)))))
expect_true(max(abs(sm - ref)) < 1e-6)

# layer norm (biased variance)
ln <- as.array(yq_layer_norm(x, eps = 1e-6))
ref <- t(apply(x_r, 1, function(r) {
    mu <- mean(r)
    v <- mean((r - mu)^2)
    (r - mu) / sqrt(v + 1e-6)
}))
expect_true(max(abs(ln - ref)) < 1e-5)

# rms norm with weight
w_r <- runif(8, 0.5, 1.5)
w <- anvl::nv_array(w_r, dtype = "f32")
rn <- as.array(yq_rms_norm(x, w, eps = 1e-6))
ref <- t(apply(x_r, 1, function(r) r / sqrt(mean(r^2) + 1e-6))) *
matrix(w_r, 6, 8, byrow = TRUE)
expect_true(max(abs(rn - ref)) < 1e-5)

# silu
sl <- as.array(yq_silu(x))
expect_true(max(abs(sl - (x_r * plogis(x_r)))) < 1e-6)

# linear on a 3-d input
w2_r <- matrix(rnorm(8 * 4), 8, 4)
x3_r <- array(rnorm(2 * 3 * 8), c(2, 3, 8))
y <- as.array(yq_linear(anvl::nv_array(x3_r, dtype = "f32"),
                        anvl::nv_array(w2_r, dtype = "f32")))
ref <- array(NA_real_, c(2, 3, 4))
for (i in 1:2) {
    for (j in 1:3) {
        ref[i, j, ] <- x3_r[i, j, ] %*% w2_r
    }
}
expect_true(max(abs(y - ref)) < 1e-5)

# slice along last dim
sl2 <- as.array(yq_slice_lastdim(x, 3L, 5L))
expect_equal(dim(sl2), c(6L, 3L))
expect_true(max(abs(sl2 - x_r[, 3:5])) < 1e-7)

# softplus (stable form vs log1p(exp))
sp <- as.array(yq_softplus(x))
expect_true(max(abs(sp - log1p(exp(x_r)))) < 1e-6)

# mish
mi <- as.array(yq_mish(x))
expect_true(max(abs(mi - x_r * tanh(log1p(exp(x_r))))) < 1e-6)

# elu (alpha = 1 and alpha = 0.5)
el <- as.array(yq_elu(x))
expect_true(max(abs(el - ifelse(x_r > 0, x_r, exp(x_r) - 1))) < 1e-6)
el2 <- as.array(yq_elu(x, alpha = 0.5))
expect_true(max(abs(el2 - ifelse(x_r > 0, x_r, 0.5 * (exp(x_r) - 1)))) < 1e-6)

# snake, per-channel alpha/beta over the last dim
a_r <- runif(8, 0.5, 1.5)
bta_r <- runif(8, 0.5, 1.5)
a <- anvl::nv_array(matrix(a_r, 1, 8), dtype = "f32")
bta <- anvl::nv_array(matrix(bta_r, 1, 8), dtype = "f32")
amat <- matrix(a_r, 6, 8, byrow = TRUE)
bmat <- matrix(bta_r, 6, 8, byrow = TRUE)
sk <- as.array(yq_snake(x, a, bta, eps = 1e-9))
expect_true(max(abs(sk - (x_r + sin(x_r * amat)^2 / (bmat + 1e-9)))) < 1e-5)
# beta = alpha default
sk1 <- as.array(yq_snake(x, a))
expect_true(max(abs(sk1 - (x_r + sin(x_r * amat)^2 / (amat + 1e-9)))) < 1e-5)

# embedding lookup (host-side gather), 0-based ids
emb <- matrix(rnorm(10 * 4), 10, 4)
ids_v <- c(0L, 3L, 9L, 3L)
ev <- as.array(yq_embedding(emb, ids_v))
expect_equal(dim(ev), c(4L, 4L))
expect_true(max(abs(ev - emb[ids_v + 1L, ])) < 1e-6)
# matrix ids -> [B, S, dim]
ids_m <- matrix(c(0L, 1L, 2L, 9L, 8L, 7L), nrow = 2, byrow = TRUE)
em <- as.array(yq_embedding(emb, ids_m))
expect_equal(dim(em), c(2L, 3L, 4L))
for (i in 1:2) {
    for (j in 1:3) {
        expect_true(max(abs(em[i, j, ] - emb[ids_m[i, j] + 1L, ])) < 1e-6)
    }
}
