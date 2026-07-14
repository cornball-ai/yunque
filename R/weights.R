#' Open a safetensors file for partial reads
#'
#' Returns a handle for \code{\link{st_read}}. Pair with
#' \code{\link{st_close}}.
#'
#' @param path Path to a .safetensors file.
#'
#' @return An opaque handle (a list with the open connection, parsed
#'   header, and data offset).
#'
#' @export
st_open <- function(path) {
    con <- file(path, "rb")
    header_len <- readBin(con, "integer", n = 1, size = 8, endian = "little")
    header <- jsonlite::fromJSON(readChar(con, header_len, useBytes = TRUE),
                                 simplifyVector = FALSE)
    header[["__metadata__"]] <- NULL
    list(con = con, header = header, data_start = 8 + header_len)
}

#' Open a sharded safetensors directory for partial reads
#'
#' Handles the \code{model.safetensors.index.json} +
#' \code{model-000NN-of-000MM.safetensors} layout. Returns a handle
#' whose \code{$header} spans all shards; \code{\link{st_read}}
#' dispatches each key to its shard. Close with \code{\link{st_close}}.
#'
#' @param dir Directory containing the index and shards.
#'
#' @return An opaque sharded handle.
#'
#' @export
st_open_sharded <- function(dir) {
    idx <- jsonlite::fromJSON(file.path(dir, "model.safetensors.index.json"),
                              simplifyVector = TRUE)
    wm <- idx$weight_map
    shards <- unique(unlist(wm))
    sts <- lapply(shards, function(s) st_open(file.path(dir, s)))
    names(sts) <- shards
    header <- do.call(c, lapply(sts, function(st) st$header))
    key_shard <- unlist(wm)
    structure(list(sts = sts, header = header, key_shard = key_shard,
                   sharded = TRUE), class = "sharded")
}

#' Close a safetensors handle
#'
#' @param st A handle from \code{\link{st_open}} /
#'   \code{\link{st_open_sharded}}.
#'
#' @export
st_close <- function(st) {
    if (isTRUE(st$sharded)) {
        for (s in st$sts) {
            close(s$con)
        }
    } else {
        close(st$con)
    }
}

#' Read one tensor from a safetensors handle as an R array
#'
#' F16/BF16 upcast to f32 (BF16 is the top half of an f32; F16 is
#' true IEEE half, converted per-element). Reading
#' a single key seeks straight to its bytes, so pulling a few tensors out
#' of a multi-GB checkpoint is cheap.
#'
#' @param st A handle from \code{\link{st_open}} /
#'   \code{\link{st_open_sharded}}.
#' @param key Tensor name.
#' @param transpose Logical. Return 2-D tensors transposed, which is free
#'   — a row-major \code{[out, in]} checkpoint matrix read column-major
#'   already IS the \code{[in, out]} logical transpose (what a linear
#'   layer wants).
#'
#' @return An R array / vector of doubles (exact f32 values).
#'
#' @export
st_read <- function(st, key, transpose = FALSE) {
    if (isTRUE(st$sharded)) {
        sub <- st$sts[[st$key_shard[[key]]]]
        return(st_read(sub, key, transpose = transpose))
    }
    meta <- st$header[[key]]
    if (is.null(meta)) {
        stop("key not in file: ", key)
    }
    shape <- as.integer(unlist(meta$shape))
    n <- prod(shape)
    seek(st$con, st$data_start + meta$data_offsets[[1]])
    vals <- switch(as.character(meta$dtype),
                   F32 = readBin(st$con, "numeric", n = n, size = 4L, endian = "little"),
                   BF16 = {
        b <- readBin(st$con, "raw", n = n * 2L)
        raw4 <- raw(n * 4L) ; idx <- seq_len(n)
        raw4[4L * idx - 1L] <- b[2L * idx - 1L]
        raw4[4L * idx] <- b[2L * idx]
        readBin(raw4, "numeric", n = n, size = 4L, endian = "little")
    },
                   F16 = .half_to_float(
                                        readBin(st$con, "integer", n = n, size = 2L, signed = FALSE,
                endian = "little")),
                   F8_E4M3 = .fp8e4m3_to_float(
            readBin(st$con, "integer", n = n, size = 1L, signed = FALSE,
                    endian = "little")),
                   F8_E5M2 = .fp8e5m2_to_float(
            readBin(st$con, "integer", n = n, size = 1L, signed = FALSE,
                    endian = "little")),
                   stop("unsupported dtype ", meta$dtype, " for ", key))
    if (length(shape) <= 1L) {
        vals
    } else if (length(shape) == 2L) {
        m <- matrix(vals, nrow = shape[2L], ncol = shape[1L]) # transpose
        if (transpose) {
            m
        } else {
            t(m)
        }
    } else {
        aperm(array(vals, dim = rev(shape)), rev(seq_along(shape)))
    }
}

# Vectorized float8 E4M3FN -> double. Input: uint8 codes.
# sign(1) exp(4, bias 7) mantissa(3); finite variant (no inf; the single
# NaN code is exp=15,mant=7). Max normal 448.
.fp8e4m3_to_float <- function(b) {
    sign <- ifelse(bitwShiftR(b, 7L) == 1L, -1, 1)
    exp <- bitwAnd(bitwShiftR(b, 3L), 0xfL)
    mant <- bitwAnd(b, 0x7L)
    val <- numeric(length(b))
    norm <- exp > 0L
    val[norm] <- (1 + mant[norm] / 8) * 2 ^ (exp[norm] - 7L)
    sub <- exp == 0L
    val[sub] <- (mant[sub] / 8) * 2 ^ (-6)
    val[exp == 15L & mant == 7L] <- NaN
    sign * val
}

# Vectorized float8 E5M2 -> double. Input: uint8 codes.
# sign(1) exp(5, bias 15) mantissa(2); has inf/nan like IEEE.
.fp8e5m2_to_float <- function(b) {
    sign <- ifelse(bitwShiftR(b, 7L) == 1L, -1, 1)
    exp <- bitwAnd(bitwShiftR(b, 2L), 0x1fL)
    mant <- bitwAnd(b, 0x3L)
    val <- numeric(length(b))
    norm <- exp > 0L & exp < 31L
    val[norm] <- (1 + mant[norm] / 4) * 2 ^ (exp[norm] - 15L)
    sub <- exp == 0L
    val[sub] <- (mant[sub] / 4) * 2 ^ (-14)
    val[exp == 31L & mant == 0L] <- Inf
    val[exp == 31L & mant != 0L] <- NaN
    sign * val
}

# Vectorized IEEE-754 half (F16) -> double. Input: uint16 codes.
# F16 is sign(1) exp(5, bias 15) mantissa(10); handles normal,
# subnormal, and inf/nan.
.half_to_float <- function(h) {
    sign <- ifelse(bitwShiftR(h, 15L) == 1L, -1, 1)
    exp <- bitwAnd(bitwShiftR(h, 10L), 0x1fL)
    mant <- bitwAnd(h, 0x3ffL)
    val <- numeric(length(h))
    norm <- exp > 0L & exp < 31L
    val[norm] <- (1 + mant[norm] / 1024) * 2 ^ (exp[norm] - 15L)
    sub <- exp == 0L
    val[sub] <- (mant[sub] / 1024) * 2 ^ (-14)
    val[exp == 31L & mant == 0L] <- Inf
    val[exp == 31L & mant != 0L] <- NaN
    sign * val
}

#' Read safetensors tensors as R arrays (base R, partial reads)
#'
#' Convenience wrapper over \code{\link{st_read}}: F16/BF16/F32
#' payloads, no torch dependency, reads only the requested keys.
#'
#' @param path Path to a .safetensors file.
#' @param keys Character vector of tensor names, or NULL for all.
#' @param transpose_2d Logical. Return 2-D tensors transposed (free; a
#'   \code{[out, in]} checkpoint matrix becomes the \code{[in, out]}
#'   layout \code{\link{linear}} wants).
#'
#' @return Named list of R arrays/vectors (doubles, exact f32 values).
#'
#' @export
read_safetensors <- function(path, keys = NULL, transpose_2d = FALSE) {
    st <- st_open(path)
    on.exit(close(st$con))
    if (is.null(keys)) {
        keys <- names(st$header)
    }
    out <- lapply(keys, function(k) st_read(st, k, transpose_2d))
    names(out) <- keys
    out
}
