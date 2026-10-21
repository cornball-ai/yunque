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
silu <- function(x) {
    x * anvl::nv_plogis(x)
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
softplus <- function(x) {
    anvl::nv_pmax(x, 0) +
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
mish <- function(x) {
    x * anvl::nv_tanh(softplus(x))
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
elu <- function(x, alpha = 1) {
    anvl::nv_pmax(x, 0) + (anvl::nv_exp(anvl::nv_pmin(x, 0)) - 1) * alpha
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
snake <- function(x, alpha, beta = NULL, eps = 1e-9) {
    s <- anvl::shape(x)
    if (is.null(beta)) {
        beta <- alpha
    }
    a <- .bcast_to(alpha, s)
    b <- .bcast_to(beta, s)
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
softmax <- function(x) {
    d <- anvl::naxes(x)
    s <- anvl::shape(x)
    m <- anvl::nv_max(x, axes = d, drop = FALSE)
    e <- anvl::nv_exp(x - anvl::nv_broadcast_to(m, s))
    z <- anvl::nv_sum(e, axes = d, drop = FALSE)
    e / anvl::nv_broadcast_to(z, s)
}

#' Layer normalization over the last dimension
#'
#' Biased variance, matching \code{torch::nn_layer_norm}. With
#' \code{weight} and \code{bias} both NULL it is the affine-free form
#' (\code{elementwise_affine = FALSE}); pass a length-\code{shape(x)[naxes(x)]}
#' \code{weight} (and optional \code{bias}) for the standard affine LayerNorm
#' \code{normalized * weight + bias}.
#'
#' @param x AnvlArray.
#' @param weight AnvlArray of length \code{shape(x)[naxes(x)]}, or NULL for no
#'   scale.
#' @param bias AnvlArray of length \code{shape(x)[naxes(x)]}, or NULL for no
#'   shift.
#' @param eps Numeric. Stability epsilon.
#'
#' @export
layer_norm <- function(x, weight = NULL, bias = NULL, eps = 1e-6) {
    d <- anvl::naxes(x)
    s <- anvl::shape(x)
    n <- s[d]
    mu <- anvl::nv_sum(x, axes = d, drop = FALSE) / n
    xc <- x - anvl::nv_broadcast_to(mu, s)
    v <- anvl::nv_sum(xc * xc, axes = d, drop = FALSE) / n
    out <- xc * anvl::nv_broadcast_to(anvl::nv_rsqrt(v + eps), s)
    if (!is.null(weight)) {
        out <- out * .bcast_to(weight, s)
    }
    if (!is.null(bias)) {
        out <- out + .bcast_to(bias, s)
    }
    out
}

#' RMS normalization over the last dimension
#'
#' Matches \code{diffuseR::ltx23_rms_norm} (float32 compute is native
#' here).
#'
#' @param x AnvlArray.
#' @param weight AnvlArray of length \code{shape(x)[naxes(x)]}, or NULL.
#' @param eps Numeric. Stability epsilon.
#'
#' @export
rms_norm <- function(x, weight = NULL, eps = 1e-6) {
    d <- anvl::naxes(x)
    s <- anvl::shape(x)
    n <- s[d]
    ms <- anvl::nv_sum(x * x, axes = d, drop = FALSE) / n
    out <- x * anvl::nv_broadcast_to(anvl::nv_rsqrt(ms + eps), s)
    if (!is.null(weight)) {
        out <- out * .bcast_to(weight, s)
    }
    out
}

# "highest" keeps CUDA dots in strict f32; TF32 measured a scaled error
# of 1.4e-03 against 2.3e-06 for "highest".
.matmul <- function(lhs, rhs, precision = "highest") {
    anvl::nv_matmul(lhs, rhs, precision = precision)
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
#'
#' @export
linear <- function(x, w_t, bias = NULL, precision = "highest") {
    s <- anvl::shape(x)
    nd <- length(s)
    d_out <- anvl::shape(w_t)[2L]
    # Merging the leading axes and splitting them back with the same
    # (column-major) order is a round trip; the last axis stays whole.
    x2 <- anvl::nv_reshape(x, c(as.integer(prod(s[-nd])), s[nd]))
    y <- .matmul(x2, w_t, precision = precision)
    out_shape <- c(s[-nd], d_out)
    if (!is.null(bias)) {
        y <- y + .bcast_to(bias, anvl::shape(y))
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
#'   \code{\link{st_read}}).
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
embedding <- function(weight, ids, zero_based = TRUE, dtype = "f32",
                      device = "cpu") {
    hidden <- ncol(weight)
    if (zero_based) {
        off <- 1L
    } else {
        off <- 0L
    }
    d <- dim(ids)
    if (is.null(d)) {
        rows <- weight[as.integer(ids) + off,, drop = FALSE]
        return(anvl::nv_array(rows, dtype = dtype, device = device))
    }
    b <- d[1L]
    s <- d[2L]
    rows <- weight[as.integer(t(ids)) + off,, drop = FALSE]
    arr <- aperm(array(t(rows), dim = c(hidden, s, b)), c(3L, 2L, 1L))
    anvl::nv_array(arr, dtype = dtype, device = device)
}

# Static slice [from, to] (1-based inclusive) along one dimension,
# keeping all others whole.
.slice_dim <- function(x, dim, from, to) {
    s <- anvl::shape(x)
    nd <- length(s)
    start <- rep(1L, nd)
    start[dim] <- as.integer(from)
    end <- as.integer(s)
    end[dim] <- as.integer(to)
    anvl::nv_static_slice(x, start_indices = start, end_indices = end,
                          strides = rep(1L, nd))
}

#' Slice a contiguous range along the last dimension
#'
#' @param x AnvlArray.
#' @param from Integer. 1-based inclusive start.
#' @param to Integer. 1-based inclusive end.
#'
#' @export
slice_lastdim <- function(x, from, to) {
    .slice_dim(x, length(anvl::shape(x)), from, to)
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
group_norm <- function(x, weight, bias, num_groups = 32L, eps = 1e-6) {
    s <- anvl::shape(x)
    b <- s[1L]; c <- s[2L]; h <- s[3L]; w <- s[4L]
    g <- as.integer(num_groups)
    per <- (c %/% g) * h * w
    xg <- .reshape(x, c(b, g, per))
    mu <- anvl::nv_sum(xg, axes = 3L, drop = FALSE) / per
    xc <- xg - anvl::nv_broadcast_to(mu, c(b, g, per))
    v <- anvl::nv_sum(xc * xc, axes = 3L, drop = FALSE) / per
    xn <- xc * anvl::nv_broadcast_to(anvl::nv_rsqrt(v + eps), c(b, g, per))
    xn <- .reshape(xn, c(b, c, h, w))
    aff <- function(p) anvl::nv_broadcast_to(anvl::nv_reshape(p,
            c(1L, c, 1L, 1L)),
        c(b, c, h, w))
    xn * aff(weight) + aff(bias)
}

#' Batch normalization (inference) over the channel dimension
#'
#' Normalizes each channel by its stored running statistics, then applies
#' an optional per-channel affine: \code{(x - mean) / sqrt(var + eps) *
#' weight + bias}. Matches \code{torch::nnf_batch_norm} in eval mode.
#' Channel is dim 2; works for any \code{[N, C, ...]} rank.
#'
#' @param x AnvlArray \code{[N, C, ...]}.
#' @param running_mean AnvlArray \code{[C]}. Per-channel running mean.
#' @param running_var AnvlArray \code{[C]}. Per-channel running variance.
#' @param weight AnvlArray \code{[C]} scale, or NULL.
#' @param bias AnvlArray \code{[C]} shift, or NULL.
#' @param eps Numeric. Stability epsilon.
#'
#' @export
batch_norm <- function(x, running_mean, running_var, weight = NULL,
                       bias = NULL, eps = 1e-5) {
    s <- anvl::shape(x)
    cshape <- rep(1L, length(s))
    cshape[2L] <- s[2L]
    bc <- function(v) anvl::nv_broadcast_to(anvl::nv_reshape(v, cshape), s)
    out <- (x - bc(running_mean)) * bc(anvl::nv_rsqrt(running_var + eps))
    if (!is.null(weight)) {
        out <- out * bc(weight)
    }
    if (!is.null(bias)) {
        out <- out + bc(bias)
    }
    out
}

#' Nearest-neighbour 2x upsampling over [B, C, H, W]
#'
#' Each pixel becomes a 2x2 block (matches
#' \code{nnf_interpolate(scale_factor = 2, mode = "nearest")}).
#'
#' @param x AnvlArray \code{[B, C, H, W]}.
#'
#' @export
upsample_nearest2d <- function(x) {
    s <- anvl::shape(x)
    b <- s[1L]; c <- s[2L]; h <- s[3L]; w <- s[4L]
    x <- anvl::nv_reshape(x, c(b, c, h, 1L, w, 1L))
    x <- anvl::nv_broadcast_to(x, c(b, c, h, 2L, w, 2L))
    .reshape(x, c(b, c, h * 2L, w * 2L))
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
slice_seq <- function(x, from, to) {
    .slice_dim(x, 2L, from, to)
}
