#' Open a safetensors file for partial reads
#'
#' Returns a handle for \code{\link{yq_st_read}}. Pair with
#' \code{\link{yq_st_close}}.
#'
#' @param path Path to a .safetensors file.
#'
#' @return An opaque handle (a list with the open connection, parsed
#'   header, and data offset).
#'
#' @export
yq_st_open <- function(path) {
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
#' whose \code{$header} spans all shards; \code{\link{yq_st_read}}
#' dispatches each key to its shard. Close with \code{\link{yq_st_close}}.
#'
#' @param dir Directory containing the index and shards.
#'
#' @return An opaque sharded handle.
#'
#' @export
yq_st_open_sharded <- function(dir) {
    idx <- jsonlite::fromJSON(file.path(dir, "model.safetensors.index.json"),
                              simplifyVector = TRUE)
    wm <- idx$weight_map
    shards <- unique(unlist(wm))
    sts <- lapply(shards, function(s) yq_st_open(file.path(dir, s)))
    names(sts) <- shards
    header <- do.call(c, lapply(sts, function(st) st$header))
    key_shard <- unlist(wm)
    structure(list(sts = sts, header = header, key_shard = key_shard,
                   sharded = TRUE), class = "yq_sharded")
}

#' Close a safetensors handle
#'
#' @param st A handle from \code{\link{yq_st_open}} /
#'   \code{\link{yq_st_open_sharded}}.
#'
#' @export
yq_st_close <- function(st) {
    if (isTRUE(st$sharded)) {
        for (s in st$sts) close(s$con)
    } else {
        close(st$con)
    }
}

#' Read one tensor from a safetensors handle as an R array
#'
#' BF16 upcasts to f32 exactly (bf16 is the top half of an f32). Reading
#' a single key seeks straight to its bytes, so pulling a few tensors out
#' of a multi-GB checkpoint is cheap.
#'
#' @param st A handle from \code{\link{yq_st_open}} /
#'   \code{\link{yq_st_open_sharded}}.
#' @param key Tensor name.
#' @param transpose Logical. Return 2-D tensors transposed, which is free
#'   — a row-major \code{[out, in]} checkpoint matrix read column-major
#'   already IS the \code{[in, out]} logical transpose (what a linear
#'   layer wants).
#'
#' @return An R array / vector of doubles (exact f32 values).
#'
#' @export
yq_st_read <- function(st, key, transpose = FALSE) {
    if (isTRUE(st$sharded)) {
        sub <- st$sts[[st$key_shard[[key]]]]
        return(yq_st_read(sub, key, transpose = transpose))
    }
    meta <- st$header[[key]]
    if (is.null(meta)) stop("key not in file: ", key)
    shape <- as.integer(unlist(meta$shape))
    n <- prod(shape)
    seek(st$con, st$data_start + meta$data_offsets[[1]])
    vals <- switch(as.character(meta$dtype),
                   F32 = readBin(st$con, "numeric", n = n, size = 4L,
                                 endian = "little"),
                   BF16 = {
        b <- readBin(st$con, "raw", n = n * 2L)
        raw4 <- raw(n * 4L); idx <- seq_len(n)
        raw4[4L * idx - 1L] <- b[2L * idx - 1L]
        raw4[4L * idx] <- b[2L * idx]
        readBin(raw4, "numeric", n = n, size = 4L, endian = "little")
    },
                   stop("unsupported dtype ", meta$dtype, " for ", key))
    if (length(shape) <= 1L) {
        vals
    } else if (length(shape) == 2L) {
        m <- matrix(vals, nrow = shape[2L], ncol = shape[1L])  # transpose
        if (transpose) m else t(m)
    } else {
        aperm(array(vals, dim = rev(shape)), rev(seq_along(shape)))
    }
}

#' Read safetensors tensors as R arrays (base R, partial reads)
#'
#' Convenience wrapper over \code{\link{yq_st_read}}: BF16/F32 payloads,
#' no torch dependency, reads only the requested keys.
#'
#' @param path Path to a .safetensors file.
#' @param keys Character vector of tensor names, or NULL for all.
#' @param transpose_2d Logical. Return 2-D tensors transposed (free; a
#'   \code{[out, in]} checkpoint matrix becomes the \code{[in, out]}
#'   layout \code{\link{yq_linear}} wants).
#'
#' @return Named list of R arrays/vectors (doubles, exact f32 values).
#'
#' @export
yq_read_safetensors <- function(path, keys = NULL, transpose_2d = FALSE) {
    st <- yq_st_open(path)
    on.exit(close(st$con))
    if (is.null(keys)) keys <- names(st$header)
    out <- lapply(keys, function(k) yq_st_read(st, k, transpose_2d))
    names(out) <- keys
    out
}
