#' Broadcasting arithmetic
#'
#' anvl's arithmetic operators broadcast scalars only: two arrays of
#' different shapes are an error, even \code{[B, S, D]} against
#' \code{[D]}. These helpers apply numpy-style broadcasting first (ranks
#' align on the right, size-1 axes expand), then run the op. anvl's own
#' \code{nv_broadcast_arrays()} aligns ranks on the left, like base R, so
#' it would pair \code{[D]} with \code{B}. Anything numpy would refuse is
#' an error here too, so a shape mismatch is caught at trace time rather
#' than guessed at. A plain R number is passed straight to the operator.
#'
#' @param x AnvlArray, or a single number.
#' @param y AnvlArray, or a single number.
#'
#' @return AnvlArray of the broadcast shape.
#'
#' @name broadcasting
NULL

.bop <- function(op, x, y) {
    if (is.numeric(x) && length(x) == 1L) {
        return(op(x, y))
    }
    if (is.numeric(y) && length(y) == 1L) {
        return(op(x, y))
    }
    s <- .broadcast_shape(anvl::shape(x), anvl::shape(y))
    op(.bcast_to(x, s), .bcast_to(y, s))
}

#' Broadcasting addition
#'
#' \code{x + y} with numpy-style broadcasting; see \link{broadcasting}.
#'
#' @param x AnvlArray, or a single number.
#' @param y AnvlArray, or a single number.
#'
#' @export
badd <- function(x, y) {
    .bop(anvl::nv_add, x, y)
}

#' Broadcasting subtraction
#'
#' \code{x - y} with numpy-style broadcasting; see \link{broadcasting}.
#'
#' @param x AnvlArray, or a single number.
#' @param y AnvlArray, or a single number.
#'
#' @export
bsub <- function(x, y) {
    .bop(anvl::nv_sub, x, y)
}

#' Broadcasting multiplication
#'
#' \code{x * y} with numpy-style broadcasting; see \link{broadcasting}.
#'
#' @param x AnvlArray, or a single number.
#' @param y AnvlArray, or a single number.
#'
#' @export
bmul <- function(x, y) {
    .bop(anvl::nv_mul, x, y)
}

#' Broadcasting division
#'
#' \code{x / y} with numpy-style broadcasting; see \link{broadcasting}.
#'
#' @param x AnvlArray, or a single number.
#' @param y AnvlArray, or a single number.
#'
#' @export
bdiv <- function(x, y) {
    .bop(anvl::nv_div, x, y)
}

# Align a lower-rank parameter p to x's shape and broadcast it there.
#
# The one case numpy's right alignment gets wrong in model code is the
# per-batch vector: shift/scale/gate [B, D] against activations
# [B, S, D], or RoPE tables [B, S, D] against heads [B, H, S, D]. Those
# are rank n-1 with a leading axis equal to x's batch axis, and the
# missing axis belongs at position 2, not on the left. Everything else
# (a [D] weight, a [1, D] row, same-rank size-1 axes) follows numpy via
# .bcast_to. The rule needs x to have at least 3 axes so a [D]
# vector against a [B, D] matrix is never mistaken for a batch vector.
.align_batched <- function(p, x) {
    s <- anvl::shape(x)
    ps <- anvl::shape(p)
    per_batch <- length(s) >= 3L && length(ps) == length(s) - 1L &&
    ps[1L] == s[1L]
    if (per_batch) {
        p <- anvl::nv_unsqueeze(p, 2L)
    }
    .bcast_to(p, s)
}
