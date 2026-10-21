# adaLN modulation, gated residuals and the timestep embedding vs base-R
# references (needs anvl + PJRT CPU).

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

nva <- function(a) anvl::nv_array(c(a), dtype = "f32", shape = dim(a))

set.seed(13)
B <- 2L
S <- 3L
D <- 4L
x_r <- array(rnorm(B * S * D), c(B, S, D))
sh_r <- matrix(rnorm(B * D), B, D)
sc_r <- matrix(rnorm(B * D), B, D)

# modulate with per-batch [B, D] shift and scale
ref_mod <- x_r
for (b in seq_len(B)) {
    for (s in seq_len(S)) {
        ref_mod[b, s, ] <- x_r[b, s, ] * (1 + sc_r[b, ]) + sh_r[b, ]
    }
}
got <- as.array(modulate(nva(x_r), nva(sh_r), nva(sc_r)))
expect_equal(dim(got), c(B, S, D))
expect_true(max(abs(got - ref_mod)) < 1e-6)

# pre-aligned [B, 1, D] inputs give the same answer
got2 <- as.array(modulate(nva(x_r), nva(array(sh_r, c(B, 1L, D))),
                          nva(array(sc_r, c(B, 1L, D)))))
expect_true(max(abs(got2 - ref_mod)) < 1e-6)

# under jit
f <- anvl::jit(function(x, sh, sc) modulate(x, sh, sc))
expect_true(max(abs(as.array(f(nva(x_r), nva(sh_r), nva(sc_r))) - ref_mod)) <
            1e-6)

# gated residual
h_r <- array(rnorm(B * S * D), c(B, S, D))
gt_r <- matrix(rnorm(B * D), B, D)
y_r <- array(rnorm(B * S * D), c(B, S, D))
ref_gr <- h_r
for (b in seq_len(B)) {
    for (s in seq_len(S)) {
        ref_gr[b, s, ] <- h_r[b, s, ] + gt_r[b, ] * y_r[b, s, ]
    }
}
got <- as.array(gated_residual(nva(h_r), nva(gt_r), nva(y_r)))
expect_true(max(abs(got - ref_gr)) < 1e-6)

# timestep embedding vs the diffusers formula
ref_emb <- function(ts, dim, flip, shift, max_period = 10000, scale = 1) {
    half <- dim %/% 2
    fr <- exp(-log(max_period) * (0:(half - 1)) / (half - shift))
    ang <- outer(ts, fr) * scale
    e <- if (flip) cbind(cos(ang), sin(ang)) else cbind(sin(ang), cos(ang))
    if (dim %% 2 == 1) {
        e <- cbind(e, 0)
    }
    e
}
ts <- c(0, 1, 500, 999)

# DDPM default: sin first, shift 1; numeric input
got <- as.array(timestep_embedding(ts, 8L))
expect_equal(dim(got), c(4L, 8L))
expect_true(max(abs(got - ref_emb(ts, 8, FALSE, 1))) < 1e-3)

# FLUX settings: cos first, shift 0, dim 256; AnvlArray input
got <- as.array(timestep_embedding(anvl::nv_array(ts, dtype = "f32"), 256L,
                                   flip_sin_to_cos = TRUE,
                                   downscale_freq_shift = 0))
expect_equal(dim(got), c(4L, 256L))
expect_true(max(abs(got - ref_emb(ts, 256, TRUE, 0))) < 1e-3)

# odd dim pads one zero column; scale multiplies the angles
got <- as.array(timestep_embedding(ts, 7L, scale = 2))
expect_equal(dim(got), c(4L, 7L))
expect_true(all(got[, 7] == 0))
expect_true(max(abs(got - ref_emb(ts, 7, FALSE, 1, scale = 2))) < 1e-3)

# under jit the frequencies are constants and the trig traces
g <- anvl::jit(function(t) timestep_embedding(t, 8L))
expect_true(max(abs(as.array(g(anvl::nv_array(ts, dtype = "f32"))) -
                    ref_emb(ts, 8, FALSE, 1))) < 1e-3)
