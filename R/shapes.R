#' Pad an axis up to a multiple
#'
#' XLA compiles one executable per input shape, so a model fed variable
#' sequence lengths recompiles at every new length. Padding the sequence
#' axis up to the next multiple of \code{multiple} caps the distinct
#' shapes, and compilations, at one per bucket. Pair it with
#' \code{\link{seq_mask}} so padded positions neither attend nor get
#' attended to. The padded length is \code{shape(out)[dim]}.
#'
#' @param x AnvlArray.
#' @param multiple Integer. Bucket size.
#' @param dim Integer. Axis to pad; default 2, the sequence axis of
#'   \code{[B, S, ...]}.
#' @param value Numeric. Fill value.
#'
#' @return AnvlArray, or \code{x} itself when its length is already a
#'   multiple.
#'
#' @export
pad_to <- function(x, multiple, dim = 2L, value = 0) {
    s <- anvl::shape(x)
    n <- s[dim]
    target <- as.integer(ceiling(n / multiple) * multiple)
    if (target == n) {
        return(x)
    }
    low <- rep(0L, length(s))
    high <- low
    high[dim] <- target - n
    anvl::nv_pad(x, value, low, high)
}

#' Additive padding mask from sequence lengths
#'
#' Builds the \code{[B, 1, 1, S]} additive bias \code{\link{sdpa}} takes:
#' 0 at valid key positions and \code{value}, a large negative number, at
#' padded ones, so padded keys get no attention weight. Lengths are
#' host-side integers, so the mask is built in R and uploaded once; inside
#' \code{anvl::jit()} pass it in as an input.
#'
#' @param lengths Integer vector \code{[B]} of valid lengths.
#' @param len Integer. Padded sequence length \code{S}.
#' @param value Numeric. Bias at padded positions. The default suits f32;
#'   use \code{-65504} for f16.
#' @param dtype Character.
#' @param device Character.
#'
#' @return AnvlArray \code{[B, 1, 1, S]}.
#'
#' @export
seq_mask <- function(lengths, len, value = -1e9, dtype = "f32",
                     device = "cpu") {
    lengths <- as.integer(lengths)
    b <- length(lengths)
    m <- matrix(value, b, len)
    for (i in seq_len(b)) {
        valid <- min(lengths[i], len)
        if (valid > 0L) {
            m[i, seq_len(valid)] <- 0
        }
    }
    anvl::nv_array(array(m, dim = c(b, 1L, 1L, len)), dtype = dtype,
                   device = device)
}

#' Additive causal mask
#'
#' A \code{[1, 1, S, S]} bias with \code{value} above the diagonal, so
#' query \code{i} attends keys \code{1..i} only. It broadcasts over batch
#' and heads in \code{\link{sdpa}}; add a \code{\link{seq_mask}} to it with
#' \code{\link{badd}} for padded causal batches.
#'
#' @param len Integer. Sequence length \code{S}.
#' @param value Numeric. Bias at masked positions.
#' @param dtype Character.
#' @param device Character.
#'
#' @return AnvlArray \code{[1, 1, S, S]}.
#'
#' @export
causal_mask <- function(len, value = -1e9, dtype = "f32", device = "cpu") {
    m <- matrix(0, len, len)
    m[upper.tri(m)] <- value
    anvl::nv_array(array(m, dim = c(1L, 1L, len, len)), dtype = dtype,
                   device = device)
}
