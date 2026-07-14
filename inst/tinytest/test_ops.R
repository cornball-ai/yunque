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

# layer norm with affine weight + bias
lw_r <- runif(8, 0.5, 1.5)
lb_r <- rnorm(8)
lw <- anvl::nv_array(lw_r, dtype = "f32")
lb <- anvl::nv_array(lb_r, dtype = "f32")
lna <- as.array(yq_layer_norm(x, weight = lw, bias = lb, eps = 1e-6))
ref <- t(apply(x_r, 1, function(r) {
    mu <- mean(r)
    v <- mean((r - mu)^2)
    (r - mu) / sqrt(v + 1e-6)
})) * matrix(lw_r, 6, 8, byrow = TRUE) + matrix(lb_r, 6, 8, byrow = TRUE)
expect_true(max(abs(lna - ref)) < 1e-5)

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
