#' Core neural-network ops for anvl
#'
#' anvl binary operators broadcast scalars only; every reduction that
#' feeds a binary op needs an explicit \code{nv_broadcast_to}. These
#' helpers package the broadcast dance behind torch-like semantics.
#' All functions work on \code{AnvlArray}s and inside \code{anvl::jit()}.
#'
#' @name ops
NULL

#' SiLU activation
#'
#' \code{x * sigmoid(x)}.
#'
#' @param x AnvlArray.
#'
#' @export
yq_silu <- function(x) {
    x * anvl::nv_logistic(x)
}

#' Softplus activation
#'
#' \code{log(1 + exp(x))}, computed in the numerically stable form
#' \code{max(x, 0) + log1p(exp(-|x|))} (matches \code{torch::nnf_softplus}
#' with the default \code{beta = 1}).
#'
#' @param x AnvlArray.
#'
#' @export
yq_softplus <- function(x) {
    anvl::nv_max(x, 0) +
        anvl::nv_log1p(anvl::nv_exp(anvl::nv_negate(anvl::nv_abs(x))))
}

#' Mish activation
#'
#' \code{x * tanh(softplus(x))} (Misra 2019); used by chatterbox's CFM
#' decoder.
#'
#' @param x AnvlArray.
#'
#' @export
yq_mish <- function(x) {
    x * anvl::nv_tanh(yq_softplus(x))
}

#' ELU activation
#'
#' \code{x} for \code{x > 0}, else \code{alpha * (exp(x) - 1)}. Written
#' branch-free as \code{max(x, 0) + alpha * (exp(min(x, 0)) - 1)} to avoid a
#' predicated select. Matches \code{torch::nnf_elu}.
#'
#' @param x AnvlArray.
#' @param alpha Numeric. Negative-saturation scale (default 1).
#'
#' @export
yq_elu <- function(x, alpha = 1) {
    anvl::nv_max(x, 0) + (anvl::nv_exp(anvl::nv_min(x, 0)) - 1) * alpha
}

#' Snake activation
#'
#' \code{x + sin(alpha * x)^2 / (beta + eps)} (Ziyin et al. 2020). The
#' single-parameter Snake HiFiGAN uses sets \code{beta = alpha}; the
#' SnakeBeta variant (BigVGAN / LTX vocoders) passes an independent
#' \code{beta}. \code{alpha}/\code{beta} must be broadcastable to \code{x}
#' (same rank, size-1 dims expand) -- reshape a per-channel \code{[C]}
#' parameter to \code{[1, C, 1]} before calling.
#'
#' @param x AnvlArray.
#' @param alpha AnvlArray. Frequency parameter, broadcastable to \code{x}.
#' @param beta AnvlArray or NULL. Magnitude parameter; NULL reuses
#'   \code{alpha}.
#' @param eps Numeric. Guards the reciprocal against division by zero.
#'
#' @export
yq_snake <- function(x, alpha, beta = NULL, eps = 1e-9) {
    s <- anvl::shape(x)
    if (is.null(beta)) beta <- alpha
    a <- anvl::nv_broadcast_to(alpha, s)
    b <- anvl::nv_broadcast_to(beta, s)
    sn <- anvl::nv_sin(x * a)
    x + sn * sn / (b + eps)
}

#' Softmax over the last dimension
#'
#' Max-subtracted for stability.
#'
#' @param x AnvlArray.
#'
#' @export
yq_softmax <- function(x) {
    d <- anvl::ndims(x)
    s <- anvl::shape(x)
    m <- anvl::nv_reduce_max(x, dims = d, drop = FALSE)
    e <- anvl::nv_exp(x - anvl::nv_broadcast_to(m, s))
    z <- anvl::nv_reduce_sum(e, dims = d, drop = FALSE)
    e / anvl::nv_broadcast_to(z, s)
}

#' Layer normalization (no affine) over the last dimension
#'
#' Biased variance, matching \code{torch::nn_layer_norm} with
#' \code{elementwise_affine = FALSE}.
#'
#' @param x AnvlArray.
#' @param eps Numeric. Stability epsilon.
#'
#' @export
yq_layer_norm <- function(x, eps = 1e-6) {
    d <- anvl::ndims(x)
    s <- anvl::shape(x)
    n <- s[d]
    mu <- anvl::nv_reduce_sum(x, dims = d, drop = FALSE) / n
    xc <- x - anvl::nv_broadcast_to(mu, s)
    v <- anvl::nv_reduce_sum(xc * xc, dims = d, drop = FALSE) / n
    xc * anvl::nv_broadcast_to(anvl::nv_rsqrt(v + eps), s)
}

#' RMS normalization over the last dimension
#'
#' Matches \code{diffuseR::ltx23_rms_norm} (float32 compute is native
#' here).
#'
#' @param x AnvlArray.
#' @param weight AnvlArray of length \code{shape(x)[ndims(x)]}, or NULL.
#' @param eps Numeric. Stability epsilon.
#'
#' @export
yq_rms_norm <- function(x, weight = NULL, eps = 1e-6) {
    d <- anvl::ndims(x)
    s <- anvl::shape(x)
    n <- s[d]
    ms <- anvl::nv_reduce_sum(x * x, dims = d, drop = FALSE) / n
    out <- x * anvl::nv_broadcast_to(anvl::nv_rsqrt(ms + eps), s)
    if (!is.null(weight)) {
        out <- out * anvl::nv_broadcast_to(weight, s)
    }
    out
}

# anvl 0.3.0's nv_matmul has no precision parameter and its CUDA dots
# run TF32 (measured scaled err 1.4e-03); the dev branch adds
# precision = with default "highest" (measured 2.3e-06, matching strict
# f32). Pass precision only where supported so yunque works on both.
.yq_matmul <- function(lhs, rhs, precision = "highest") {
    if ("precision" %in% names(formals(anvl::nv_matmul))) {
        anvl::nv_matmul(lhs, rhs, precision = precision)
    } else {
        anvl::nv_matmul(lhs, rhs)
    }
}

#' Linear layer (bias-free)
#'
#' \code{x [..., d_in] \%*\% w_t [d_in, d_out]}. Pass the weight
#' pre-transposed (torch checkpoints store \code{[d_out, d_in]}); do the
#' transpose once at load time, not per call. Higher-rank inputs are
#' flattened to 2-D for the matmul and restored after.
#'
#' @param x AnvlArray \code{[..., d_in]}.
#' @param w_t AnvlArray \code{[d_in, d_out]}.
#' @param bias AnvlArray \code{[d_out]} or NULL.
#' @param precision Character. Matmul precision; the \code{"highest"}
#'   default forbids TF32-style downgrades so f32 is honest f32.
#'   Ignored on anvl 0.3.0, whose \code{nv_matmul} has no precision
#'   parameter (and whose CUDA dots run TF32 regardless).
#'
#' @export
yq_linear <- function(x, w_t, bias = NULL, precision = "highest") {
    s <- anvl::shape(x)
    nd <- length(s)
    d_out <- anvl::shape(w_t)[2L]
    x2 <- anvl::nv_reshape(x, c(as.integer(prod(s[-nd])), s[nd]))
    y <- .yq_matmul(x2, w_t, precision = precision)
    out_shape <- c(s[-nd], d_out)
    if (!is.null(bias)) {
        y <- y + anvl::nv_broadcast_to(bias, anvl::shape(y))
    }
    anvl::nv_reshape(y, out_shape)
}

#' Embedding lookup (host-side gather)
#'
#' Gathers rows of an embedding table for the given token ids. The table
#' stays an R matrix (never resident on device); only the gathered result
#' crosses to anvl -- ids are host-side integers at the input boundary, so
#' this sidesteps an on-device gather entirely (the pattern the anvl model
#' ports use for token/position embeddings).
#'
#' @param weight R matrix \code{[num_embeddings, dim]} (e.g. from
#'   \code{\link{yq_st_read}}).
#' @param ids Integer vector \code{[N]} or matrix \code{[B, S]} of token ids.
#' @param zero_based Logical. TRUE (default) treats \code{ids} as 0-based
#'   (torch convention); FALSE as 1-based R indices.
#' @param dtype Character. Output dtype.
#' @param device Character.
#'
#' @return AnvlArray \code{[N, dim]} for a vector \code{ids}, or
#'   \code{[B, S, dim]} for a matrix.
#'
#' @export
yq_embedding <- function(weight, ids, zero_based = TRUE, dtype = "f32",
                         device = "cpu") {
    hidden <- ncol(weight)
    off <- if (zero_based) 1L else 0L
    d <- dim(ids)
    if (is.null(d)) {
        rows <- weight[as.integer(ids) + off, , drop = FALSE]
        return(anvl::nv_array(rows, dtype = dtype, device = device))
    }
    b <- d[1L]
    s <- d[2L]
    rows <- weight[as.integer(t(ids)) + off, , drop = FALSE]
    arr <- aperm(array(t(rows), dim = c(hidden, s, b)), c(3L, 2L, 1L))
    anvl::nv_array(arr, dtype = dtype, device = device)
}

# Static slice [from, to] (1-based inclusive) along one dimension,
# keeping all others whole.
.yq_slice_dim <- function(x, dim, from, to) {
    s <- anvl::shape(x)
    nd <- length(s)
    start <- rep(1L, nd)
    start[dim] <- as.integer(from)
    limit <- as.integer(s)
    limit[dim] <- as.integer(to)
    anvl::nv_static_slice(x, start_indices = start, limit_indices = limit,
                          strides = rep(1L, nd))
}

#' Slice a contiguous range along the last dimension
#'
#' @param x AnvlArray.
#' @param from Integer. 1-based inclusive start.
#' @param to Integer. 1-based inclusive end.
#'
#' @export
yq_slice_lastdim <- function(x, from, to) {
    .yq_slice_dim(x, length(anvl::shape(x)), from, to)
}

#' Group normalization over [B, C, H, W]
#'
#' Normalizes within \code{num_groups} channel groups (over the group's
#' channels and all spatial positions), then applies a per-channel
#' affine. Matches \code{torch::nn_group_norm}.
#'
#' @param x AnvlArray \code{[B, C, H, W]}.
#' @param weight AnvlArray \code{[C]} scale.
#' @param bias AnvlArray \code{[C]} shift.
#' @param num_groups Integer.
#' @param eps Numeric.
#'
#' @export
yq_group_norm <- function(x, weight, bias, num_groups = 32L, eps = 1e-6) {
    s <- anvl::shape(x)
    b <- s[1L]; c <- s[2L]; h <- s[3L]; w <- s[4L]
    g <- as.integer(num_groups)
    per <- (c %/% g) * h * w
    xg <- anvl::nv_reshape(x, c(b, g, per))
    mu <- anvl::nv_reduce_sum(xg, dims = 3L, drop = FALSE) / per
    xc <- xg - anvl::nv_broadcast_to(mu, c(b, g, per))
    v <- anvl::nv_reduce_sum(xc * xc, dims = 3L, drop = FALSE) / per
    xn <- xc * anvl::nv_broadcast_to(anvl::nv_rsqrt(v + eps), c(b, g, per))
    xn <- anvl::nv_reshape(xn, c(b, c, h, w))
    aff <- function(p) anvl::nv_broadcast_to(anvl::nv_reshape(p, c(1L, c, 1L, 1L)),
                                             c(b, c, h, w))
    xn * aff(weight) + aff(bias)
}

#' Nearest-neighbour 2x upsampling over [B, C, H, W]
#'
#' Each pixel becomes a 2x2 block (matches
#' \code{nnf_interpolate(scale_factor = 2, mode = "nearest")}).
#'
#' @param x AnvlArray \code{[B, C, H, W]}.
#'
#' @export
yq_upsample_nearest2d <- function(x) {
    s <- anvl::shape(x)
    b <- s[1L]; c <- s[2L]; h <- s[3L]; w <- s[4L]
    x <- anvl::nv_reshape(x, c(b, c, h, 1L, w, 1L))
    x <- anvl::nv_broadcast_to(x, c(b, c, h, 2L, w, 2L))
    anvl::nv_reshape(x, c(b, c, h * 2L, w * 2L))
}

#' Slice a contiguous range along the sequence dimension (dim 2)
#'
#' For \code{[B, S, D]} tensors: split text/image token spans of a
#' concatenated sequence.
#'
#' @param x AnvlArray \code{[B, S, ...]}.
#' @param from Integer. 1-based inclusive start.
#' @param to Integer. 1-based inclusive end.
#'
#' @export
yq_slice_seq <- function(x, from, to) {
    .yq_slice_dim(x, 2L, from, to)
}
