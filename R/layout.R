# Torch layout semantics on anvl.
#
# anvl 0.5 follows base R: nv_reshape() is column-major and
# nv_broadcast_to() aligns ranks on the left. Ported torch code expects
# row-major reshapes and numpy's right-aligned broadcasting, so yunque
# goes through these helpers wherever the two differ.

# Row-major (torch) reshape. Reversing the axes turns row-major order
# into column-major order, so reshape the reversed array to the reversed
# shape and reverse back. A reshape that only adds or drops size-1 axes
# moves no elements and skips the transposes.
.reshape <- function(x, shape) {
    shape <- as.integer(shape)
    s <- anvl::shape(x)
    if (identical(s[s != 1L], shape[shape != 1L])) {
        return(anvl::nv_reshape(x, shape))
    }
    n <- length(s)
    if (n > 1L) {
        x <- anvl::nv_transpose(x, rev(seq_len(n)))
    }
    x <- anvl::nv_reshape(x, rev(shape))
    m <- length(shape)
    if (m > 1L) {
        x <- anvl::nv_transpose(x, rev(seq_len(m)))
    }
    x
}

# numpy's broadcast shape of two shapes: right-align, then each axis pair
# must match or include a 1.
.broadcast_shape <- function(a, b) {
    n <- max(length(a), length(b))
    a <- c(rep(1L, n - length(a)), a)
    b <- c(rep(1L, n - length(b)), b)
    ok <- a == b | a == 1L | b == 1L
    if (!all(ok)) {
        stop(sprintf("Shapes (%s) and (%s) are not broadcastable.",
                     paste(a, collapse = "x"), paste(b, collapse = "x")),
             call. = FALSE)
    }
    as.integer(pmax(a, b))
}

# Broadcast x to shape the numpy way: missing axes are added on the left,
# then size-1 axes expand. Inserting size-1 axes moves no elements, so a
# plain reshape pads the rank before nv_broadcast_to().
.bcast_to <- function(x, shape) {
    shape <- as.integer(shape)
    s <- anvl::shape(x)
    k <- length(shape) - length(s)
    if (k < 0L) {
        stop(sprintf("Cannot broadcast (%s) to lower rank (%s).",
                     paste(s, collapse = "x"), paste(shape, collapse = "x")),
             call. = FALSE)
    }
    if (k > 0L) {
        s <- c(rep(1L, k), s)
        x <- anvl::nv_reshape(x, s)
    }
    if (identical(s, shape)) {
        return(x)
    }
    anvl::nv_broadcast_to(x, shape)
}
