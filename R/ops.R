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
