# Conv family vs a base-R cross-correlation reference (needs anvl + PJRT CPU).

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

nva <- function(a) anvl::nv_array(c(a), dtype = "f32", shape = dim(a))

# base-R N-D cross-correlation, NCHW-style, zero pad, stride/dilation/groups
ref_conv <- function(x, w, stride, padding, dilation, groups) {
    xd <- dim(x)
    wd <- dim(w)
    N <- xd[1]
    sp <- xd[-(1:2)]
    Cout <- wd[1]
    cin_g <- wd[2]
    ks <- wd[-(1:2)]
    k <- length(sp)
    stride <- rep_len(stride, k)
    padding <- rep_len(padding, k)
    dilation <- rep_len(dilation, k)
    out_sp <- (sp + 2 * padding - dilation * (ks - 1) - 1) %/% stride + 1
    out <- array(0, dim = c(N, Cout, out_sp))
    co_per_g <- Cout %/% groups
    taps <- expand.grid(lapply(ks, seq_len))
    ipos <- expand.grid(lapply(out_sp, seq_len))
    for (n in seq_len(N)) {
        for (co in seq_len(Cout)) {
            g <- (co - 1) %/% co_per_g
            for (oi in seq_len(nrow(ipos))) {
                o <- as.integer(ipos[oi, ])
                acc <- 0
                for (ci in seq_len(cin_g)) {
                    for (ti in seq_len(nrow(taps))) {
                        tp <- as.integer(taps[ti, ])
                        inpos <- (o - 1) * stride + (tp - 1) * dilation + 1 - padding
                        if (all(inpos >= 1 & inpos <= sp)) {
                            acc <- acc + x[matrix(c(n, g * cin_g + ci, inpos), 1)] *
                                w[matrix(c(co, ci, tp), 1)]
                        }
                    }
                }
                out[matrix(c(n, co, o), 1)] <- acc
            }
        }
    }
    out
}

reltol <- function(got, ref) max(abs(got - ref)) / max(abs(ref))

set.seed(1)

# conv1d + symmetric padding + bias
x <- array(rnorm(1 * 3 * 8), c(1, 3, 8))
w <- array(rnorm(4 * 3 * 3), c(4, 3, 3))
b <- rnorm(4)
got <- as.array(conv1d(nva(x), nva(w), bias = nva(b), padding = 1))
ref <- sweep(ref_conv(x, w, 1, 1, 1, 1), 2, b, "+")
expect_true(reltol(got, ref) < 1e-3)

# conv1d grouped (depthwise), no bias
xg <- array(rnorm(1 * 4 * 6), c(1, 4, 6))
wg <- array(rnorm(4 * 1 * 3), c(4, 1, 3))
expect_true(reltol(as.array(conv1d(nva(xg), nva(wg), groups = 4)),
                   ref_conv(xg, wg, 1, 0, 1, 4)) < 1e-3)

# conv2d + stride + bias
x2 <- array(rnorm(1 * 2 * 6 * 6), c(1, 2, 6, 6))
w2 <- array(rnorm(3 * 2 * 3 * 3), c(3, 2, 3, 3))
b2 <- rnorm(3)
got2 <- as.array(conv2d(nva(x2), nva(w2), bias = nva(b2), stride = 2))
ref2 <- sweep(ref_conv(x2, w2, 2, 0, 1, 1), 2, b2, "+")
expect_true(reltol(got2, ref2) < 1e-3)

# conv3d, no bias
x3 <- array(rnorm(1 * 2 * 4 * 4 * 4), c(1, 2, 4, 4, 4))
w3 <- array(rnorm(2 * 2 * 2 * 2 * 2), c(2, 2, 2, 2, 2))
expect_true(reltol(as.array(conv3d(nva(x3), nva(w3))),
                   ref_conv(x3, w3, 1, 0, 1, 1)) < 1e-3)

# batch_norm (inference), affine
xb <- array(rnorm(2 * 3 * 5), c(2, 3, 5))
rmn <- runif(3)
rvr <- runif(3, 0.5, 1.5)
gw <- runif(3, 0.5, 1.5)
gb <- rnorm(3)
gotb <- as.array(batch_norm(
    nva(xb),
    anvl::nv_array(rmn, dtype = "f32"), anvl::nv_array(rvr, dtype = "f32"),
    weight = anvl::nv_array(gw, dtype = "f32"),
    bias = anvl::nv_array(gb, dtype = "f32"), eps = 1e-5
))
refb <- xb
for (ci in 1:3) {
    refb[, ci, ] <- (xb[, ci, ] - rmn[ci]) / sqrt(rvr[ci] + 1e-5) * gw[ci] + gb[ci]
}
expect_true(reltol(gotb, refb) < 1e-3)

# conv_transpose1d vs a base-R scatter reference (the definition of the
# adjoint conv), across stride / padding / output_padding / dilation.
ref_convt1d <- function(x, w, stride, padding, output_padding, dilation) {
    Cin <- dim(x)[2]
    L <- dim(x)[3]
    Cout <- dim(w)[2]
    k <- dim(w)[3]
    lout <- (L - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1
    out <- array(0, c(1L, Cout, as.integer(lout)))
    for (o in seq_len(Cout)) {
        for (li in seq_len(L)) {
            for (kk in seq_len(k)) {
                pos <- (li - 1) * stride + (kk - 1) * dilation - padding + 1
                if (pos >= 1 && pos <= lout) {
                    for (i in seq_len(Cin)) {
                        out[1, o, pos] <- out[1, o, pos] + x[1, i, li] * w[i, o, kk]
                    }
                }
            }
        }
    }
    out
}

xt <- array(rnorm(1 * 2 * 5), c(1, 2, 5))
wt <- array(rnorm(2 * 3 * 4), c(2, 3, 4)) # torch layout [C_in, C_out, kW]
for (cfg in list(
    list(s = 1L, p = 0L, op = 0L, d = 1L),
    list(s = 2L, p = 0L, op = 0L, d = 1L),
    list(s = 2L, p = 1L, op = 1L, d = 1L),
    list(s = 1L, p = 0L, op = 0L, d = 2L)
)) {
    got <- as.array(conv_transpose1d(
        nva(xt), nva(wt),
        stride = cfg$s, padding = cfg$p, output_padding = cfg$op, dilation = cfg$d
    ))
    ref <- ref_convt1d(xt, wt, cfg$s, cfg$p, cfg$op, cfg$d)
    expect_equal(dim(got), dim(ref))
    expect_true(reltol(got, ref) < 1e-3)
}

# conv_transpose1d bias + groups guard
expect_error(conv_transpose1d(nva(xt), nva(wt), groups = 2L), "groups = 1")
