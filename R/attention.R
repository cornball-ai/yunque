#' Scaled dot-product attention
#'
#' Composed from batched matmul and softmax; the full
#' \code{[B, H, Sq, Sk]} score matrix materializes (XLA may fuse, but
#' budget for the worst case at long sequence lengths).
#'
#' @param query AnvlArray \code{[B, H, S, D]}.
#' @param key AnvlArray \code{[B, H, S, D]}.
#' @param value AnvlArray \code{[B, H, S, D]}.
#' @param precision Character. Matmul precision; the \code{"highest"}
#'   default forbids TF32-style downgrades so f32 is honest f32.
#'   Ignored on anvl 0.3.0, whose \code{nv_matmul} has no precision
#'   parameter (and whose CUDA dots run TF32 regardless).
#'
#' @return AnvlArray \code{[B, H, Sq, D]}.
#'
#' @export
yq_sdpa <- function(query, key, value, precision = "highest") {
    nd <- anvl::ndims(query)
    d <- anvl::shape(query)[nd]
    perm <- seq_len(nd)
    perm[c(nd - 1L, nd)] <- perm[c(nd, nd - 1L)]
    scores <- .yq_matmul(query, anvl::nv_transpose(key, perm),
                         precision = precision) * (1 / sqrt(d))
    .yq_matmul(yq_softmax(scores), value, precision = precision)
}

#' Apply rotary position embeddings (interleaved pairs)
#'
#' FLUX convention: the last dimension holds \code{D/2} interleaved
#' (real, imag) pairs; \code{cos}/\code{sin} carry each angle twice so
#' they multiply elementwise. All three arguments must share the same
#' 4-D shape \code{[B, H, S, D]} — broadcast \code{cos}/\code{sin}
#' before calling.
#'
#' @param x AnvlArray \code{[B, H, S, D]}, D even.
#' @param cos AnvlArray with the same shape as \code{x}.
#' @param sin AnvlArray with the same shape as \code{x}.
#'
#' @export
yq_rope_apply <- function(x, cos, sin) {
    s <- anvl::shape(x)
    r <- anvl::nv_reshape(x, c(s[1L], s[2L], s[3L], s[4L] %/% 2L, 2L))
    xr <- r[, , , , 1L]
    xi <- r[, , , , 2L]
    rot <- anvl::nv_concatenate(
                                anvl::nv_unsqueeze(anvl::nv_negate(xi), 5L),
                                anvl::nv_unsqueeze(xr, 5L),
                                dimension = 5L
    )
    x * cos + anvl::nv_reshape(rot, s) * sin
}
