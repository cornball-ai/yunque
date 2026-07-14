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
#' @param mask AnvlArray additive bias added to the (scaled) scores,
#'   broadcast to the score shape \code{[B, H, Sq, Sk]} — a causal /
#'   padding mask, or a learned bias (e.g. T5 relative-position bias).
#'   NULL for none.
#' @param scale Numeric score scale. NULL (default) uses
#'   \code{1 / sqrt(head_dim)}; pass \code{1} for the unscaled attention
#'   T5 uses (its relative-position bias is added to raw \code{qk^T}).
#'
#' @return AnvlArray \code{[B, H, Sq, D]}.
#'
#' @export
sdpa <- function(query, key, value, mask = NULL, scale = NULL,
                    precision = "highest") {
    nd <- anvl::ndims(query)
    d <- anvl::shape(query)[nd]
    if (is.null(scale)) {
        scale <- 1 / sqrt(d)
    }
    perm <- seq_len(nd)
    perm[c(nd - 1L, nd)] <- perm[c(nd, nd - 1L)]
    scores <- .matmul(query, anvl::nv_transpose(key, perm),
                         precision = precision) * scale
    if (!is.null(mask)) {
        scores <- scores + anvl::nv_broadcast_to(mask, anvl::shape(scores))
    }
    .matmul(softmax(scores), value, precision = precision)
}

#' GELU activation
#'
#' Gaussian Error Linear Unit. The \code{"tanh"} approximation
#' (transformers' \code{gelu_new}) is what T5, Gemma, FLUX feed-forwards,
#' and most GELU transformer ports actually use; \code{"none"} is the
#' exact erf form.
#'
#' @param x AnvlArray.
#' @param approximate Character. \code{"tanh"} (default) or \code{"none"}.
#'
#' @export
gelu <- function(x, approximate = c("tanh", "none")) {
    approximate <- match.arg(approximate)
    if (approximate == "tanh") {
        inner <- (x + x * x * x * 0.044715) * sqrt(2 / pi)
        x * 0.5 * (anvl::nv_tanh(inner) + 1)
    } else {
        x * 0.5 * (anvl::nv_erf(x * (1 / sqrt(2))) + 1)
    }
}

#' Apply split-half rotary embeddings (Llama/Qwen convention)
#'
#' Rotates pairs formed by splitting the last dimension in half: element
#' i pairs with element i + D/2 (\code{rotate_half}). Distinct from the
#' interleaved-pair \code{\link{rope_apply}} used by FLUX.
#'
#' @param x AnvlArray \code{[B, H, S, D]}, D even.
#' @param cos AnvlArray broadcastable to \code{[B, H, S, D/2]}.
#' @param sin AnvlArray broadcastable to \code{[B, H, S, D/2]}.
#'
#' @export
rope_split <- function(x, cos, sin) {
    d <- anvl::shape(x)[anvl::ndims(x)]
    r <- d %/% 2L
    first <- slice_lastdim(x, 1L, r)
    second <- slice_lastdim(x, r + 1L, 2L * r)
    out_first <- first * cos - second * sin
    out_second <- second * cos + first * sin
    anvl::nv_concatenate(out_first, out_second, dimension = anvl::ndims(x))
}

#' Repeat KV heads to match query heads (grouped-query attention)
#'
#' Interleaved expansion (\code{repeat_interleave} over the head dim):
#' each KV head is repeated \code{groups} times consecutively, so head
#' \code{j} of the query maps to KV head \code{floor(j / groups)}.
#'
#' @param x AnvlArray \code{[B, KV, S, D]}.
#' @param groups Integer. Query heads per KV head.
#'
#' @return AnvlArray \code{[B, KV * groups, S, D]}.
#'
#' @export
repeat_kv <- function(x, groups) {
    if (groups == 1L) {
        return(x)
    }
    s <- anvl::shape(x)
    x <- anvl::nv_unsqueeze(x, 3L) # [B, KV, 1, S, D]
    x <- anvl::nv_broadcast_to(x, c(s[1L], s[2L], groups, s[3L], s[4L]))
    anvl::nv_reshape(x, c(s[1L], s[2L] * groups, s[3L], s[4L]))
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
rope_apply <- function(x, cos, sin) {
    s <- anvl::shape(x)
    r <- anvl::nv_reshape(x, c(s[1L], s[2L], s[3L], s[4L] %/% 2L, 2L))
    xr <- r[,,,, 1L]
    xi <- r[,,,, 2L]
    rot <- anvl::nv_concatenate(anvl::nv_unsqueeze(anvl::nv_negate(xi), 5L),
                                anvl::nv_unsqueeze(xr, 5L), dimension = 5L)
    x * cos + anvl::nv_reshape(rot, s) * sin
}
