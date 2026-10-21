#' Linear weight spec
#'
#' Describes a \code{[out, in]} checkpoint matrix to be returned in the
#' \code{[in, out]} layout \code{\link{linear}} takes. See
#' \code{\link{load_weights}}.
#'
#' @param key Character. Tensor name in the checkpoint.
#'
#' @return A weight spec.
#'
#' @export
w_linear <- function(key) {
    structure(list(key = key, transpose = TRUE), class = "yunque_weight")
}

#' Tensor weight spec
#'
#' Describes a tensor to be returned as stored: norm weights, biases,
#' conv kernels, embedding tables. See \code{\link{load_weights}}.
#'
#' @param key Character. Tensor name in the checkpoint.
#' @param transpose Logical. Transpose a 2-D tensor on read.
#'
#' @return A weight spec.
#'
#' @export
w_tensor <- function(key, transpose = FALSE) {
    structure(list(key = key, transpose = transpose), class = "yunque_weight")
}

#' Concatenated weight spec
#'
#' Describes several checkpoint tensors concatenated along \code{dim}
#' after reading, for a fused projection such as one QKV weight assembled
#' from separate q, k and v matrices. Matrices are read transposed as
#' \code{\link{w_linear}} does, so \code{dim = 2} stacks their output
#' columns; pass \code{transpose = FALSE} for vectors (biases) joined end
#' to end. See \code{\link{load_weights}}.
#'
#' @param keys Character vector of tensor names.
#' @param dim Integer. Axis to concatenate along, in the returned layout:
#'   1 or 2 for matrices, ignored for vectors.
#' @param transpose Logical. Transpose 2-D tensors on read.
#'
#' @return A weight spec.
#'
#' @export
w_cat <- function(keys, dim, transpose = TRUE) {
    structure(list(key = keys, transpose = transpose, dim = as.integer(dim)),
              class = "yunque_weight")
}

#' Per-layer weight specs
#'
#' Evaluates \code{spec(i)} for \code{i} in \code{1:n}, giving the list of
#' per-layer spec trees a stack of identical blocks needs. The function
#' builds the checkpoint keys from the index (checkpoints count from 0).
#' See \code{\link{load_weights}}.
#'
#' @param n Integer. Number of layers.
#' @param spec Function of the 1-based layer index returning a spec tree.
#'
#' @return A list of \code{n} spec trees.
#'
#' @export
w_layers <- function(n, spec) {
    lapply(seq_len(n), spec)
}

#' Load a weight tree from a safetensors checkpoint
#'
#' A port's weights are a nested named list mirroring the model: blocks,
#' then layers, then the tensors each layer needs. Instead of a loader per
#' model that opens the file, reads each key, transposes matrices and
#' uploads them, describe the tree once with \code{\link{w_linear}},
#' \code{\link{w_tensor}}, \code{\link{w_cat}} and \code{\link{w_layers}}
#' and let this walk it. Each leaf is read with \code{\link{st_read}} and
#' returned as an \code{AnvlArray} on \code{device}. Non-spec values in
#' the tree (\code{NULL}, numbers, strings) pass through, so static
#' configuration can sit next to the weights it describes.
#'
#' Every key named in the spec must exist, an error otherwise. Keys in
#' the checkpoint the spec never reads are reported according to
#' \code{unused}, so a port can prove its census: every tensor the file
#' ships has a slot. They are also attached as
#' \code{attr(result, "unused_keys")}.
#'
#' @param st A handle from \code{\link{st_open}} or
#'   \code{\link{st_open_sharded}}, or a path to a .safetensors file,
#'   opened and closed here.
#' @param spec Nested list of weight specs.
#' @param dtype Character. Device dtype for every leaf.
#' @param device Character.
#' @param unused Character. What to do about checkpoint keys the spec does
#'   not read: \code{"warn"} (default), \code{"error"} or \code{"ignore"}.
#'
#' @return The spec tree with \code{AnvlArray} leaves.
#'
#' @export
load_weights <- function(st, spec, dtype = "f32", device = "cpu",
                         unused = c("warn", "error", "ignore")) {
    unused <- match.arg(unused)
    if (is.character(st)) {
        st <- st_open(st)
        on.exit(st_close(st), add = TRUE)
    }
    used <- character()
    read_leaf <- function(w) {
        used <<- c(used, w$key)
        if (length(w$key) == 1L) {
            val <- st_read(st, w$key, transpose = w$transpose)
        } else {
            parts <- lapply(w$key, function(k) {
                st_read(st, k, transpose = w$transpose)
            })
            val <- .cat_arrays(parts, w$dim)
        }
        anvl::nv_array(val, dtype = dtype, device = device)
    }
    walk <- function(node) {
        if (inherits(node, "yunque_weight")) {
            return(read_leaf(node))
        }
        if (is.list(node)) {
            return(lapply(node, walk))
        }
        node
    }
    out <- walk(spec)
    leftover <- setdiff(names(st$header), used)
    if (length(leftover) > 0L && unused != "ignore") {
        shown <- leftover[seq_len(min(5L, length(leftover)))]
        msg <- sprintf("%d checkpoint key(s) not read by the spec, e.g. %s",
                       length(leftover), paste(shown, collapse = ", "))
        if (unused == "error") {
            stop(msg)
        }
        warning(msg)
    }
    attr(out, "unused_keys") <- leftover
    out
}

# Concatenate arrays read from a checkpoint along one axis: vectors end
# to end, matrices by row (dim 1) or column (dim 2). The fused
# projections this serves never need higher ranks.
.cat_arrays <- function(parts, dim) {
    is_vec <- vapply(parts, function(p) is.null(dim(p)), logical(1))
    if (all(is_vec)) {
        return(do.call(c, parts))
    }
    rank2 <- vapply(parts, function(p) length(dim(p)) == 2L, logical(1))
    if (!all(rank2)) {
        stop("w_cat() supports vectors and matrices only")
    }
    if (identical(dim, 1L)) {
        return(do.call(rbind, parts))
    }
    if (identical(dim, 2L)) {
        return(do.call(cbind, parts))
    }
    stop("w_cat(): dim must be 1 or 2 for matrices")
}
