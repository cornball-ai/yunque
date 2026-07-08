#' @keywords internal
.yq_st_open <- function(path) {
    con <- file(path, "rb")
    header_len <- readBin(con, "integer", n = 1, size = 8, endian = "little")
    header <- jsonlite::fromJSON(readChar(con, header_len, useBytes = TRUE),
                                 simplifyVector = FALSE)
    header[["__metadata__"]] <- NULL
    list(con = con, header = header, data_start = 8 + header_len)
}

# Open a sharded safetensors directory (model.safetensors.index.json +
# model-000NN-of-000MM.safetensors). Returns a virtual handle whose
# $header/$read dispatch to the right shard by key. .yq_st_read works
# unchanged on either handle.
.yq_st_open_sharded <- function(dir) {
    idx <- jsonlite::fromJSON(file.path(dir, "model.safetensors.index.json"),
                              simplifyVector = TRUE)
    wm <- idx$weight_map
    shards <- unique(unlist(wm))
    sts <- lapply(shards, function(s) .yq_st_open(file.path(dir, s)))
    names(sts) <- shards
    header <- do.call(c, lapply(sts, function(st) st$header))
    key_shard <- unlist(wm)
    structure(list(sts = sts, header = header, key_shard = key_shard,
                   sharded = TRUE), class = "yq_sharded")
}

.yq_st_close <- function(st) {
    if (isTRUE(st$sharded)) {
        for (s in st$sts) close(s$con)
    } else {
        close(st$con)
    }
}

# Read one tensor as an R array. BF16 upcasts to f32 exactly (bf16 is
# the top half of an f32). transpose: return 2-D tensors transposed,
# which is free — a row-major [out, in] checkpoint matrix read
# column-major already IS the [in, out] logical transpose.
.yq_st_read <- function(st, key, transpose = FALSE) {
    if (isTRUE(st$sharded)) {
        sub <- st$sts[[st$key_shard[[key]]]]
        return(.yq_st_read(sub, key, transpose = transpose))
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
#' Minimal safetensors reader supporting BF16 and F32 payloads, no torch
#' dependency. Reads only the requested keys, so pulling a few tensors
#' out of a multi-GB checkpoint is cheap.
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
    st <- .yq_st_open(path)
    on.exit(close(st$con))
    if (is.null(keys)) keys <- names(st$header)
    out <- lapply(keys, function(k) .yq_st_read(st, k, transpose_2d))
    names(out) <- keys
    out
}

#' Load FLUX.2 Klein transformer weights into an anvl pytree
#'
#' Reads every transformer weight from the checkpoint (bf16 upcast to
#' f32), transposing 2-D linears to \code{[in, out]}, and wraps each as
#' an \code{AnvlArray} on \code{device} — freeing the R copy as it goes,
#' so peak host memory stays near one tensor rather than the full 15.5
#' GB twice. Returns the nested list \code{\link{yq_flux2_transformer}}
#' expects.
#'
#' @param path Path to \code{transformer/diffusion_pytorch_model.safetensors}.
#' @param num_layers Integer. Double-stream blocks (5).
#' @param num_single_layers Integer. Single-stream blocks (20).
#' @param device Character. Target device.
#'
#' @return Weights pytree.
#'
#' @export
yq_flux2_load_weights <- function(path, num_layers = 5L,
                                  num_single_layers = 20L, device = "cpu") {
    st <- .yq_st_open(path)
    on.exit(close(st$con))
    lin <- function(key) {
        a <- anvl::nv_array(.yq_st_read(st, key, transpose = TRUE),
                            dtype = "f32", device = device)
        a
    }
    vec <- function(key) {
        anvl::nv_array(.yq_st_read(st, key), dtype = "f32", device = device)
    }

    w <- list(
        x_embedder = lin("x_embedder.weight"),
        context_embedder = lin("context_embedder.weight"),
        time_1 = lin("time_guidance_embed.timestep_embedder.linear_1.weight"),
        time_2 = lin("time_guidance_embed.timestep_embedder.linear_2.weight"),
        dsm_img = lin("double_stream_modulation_img.linear.weight"),
        dsm_txt = lin("double_stream_modulation_txt.linear.weight"),
        single_mod = lin("single_stream_modulation.linear.weight"),
        norm_out = lin("norm_out.linear.weight"),
        proj_out = lin("proj_out.weight")
    )

    w$double <- lapply(seq_len(num_layers) - 1L, function(i) {
        p <- sprintf("transformer_blocks.%d.", i)
        list(
            to_q = lin(paste0(p, "attn.to_q.weight")),
            to_k = lin(paste0(p, "attn.to_k.weight")),
            to_v = lin(paste0(p, "attn.to_v.weight")),
            norm_q = vec(paste0(p, "attn.norm_q.weight")),
            norm_k = vec(paste0(p, "attn.norm_k.weight")),
            add_q_proj = lin(paste0(p, "attn.add_q_proj.weight")),
            add_k_proj = lin(paste0(p, "attn.add_k_proj.weight")),
            add_v_proj = lin(paste0(p, "attn.add_v_proj.weight")),
            norm_added_q = vec(paste0(p, "attn.norm_added_q.weight")),
            norm_added_k = vec(paste0(p, "attn.norm_added_k.weight")),
            to_out = lin(paste0(p, "attn.to_out.0.weight")),
            to_add_out = lin(paste0(p, "attn.to_add_out.weight")),
            ff_in = lin(paste0(p, "ff.linear_in.weight")),
            ff_out = lin(paste0(p, "ff.linear_out.weight")),
            ff_context_in = lin(paste0(p, "ff_context.linear_in.weight")),
            ff_context_out = lin(paste0(p, "ff_context.linear_out.weight"))
        )
    })

    w$single <- lapply(seq_len(num_single_layers) - 1L, function(i) {
        p <- sprintf("single_transformer_blocks.%d.", i)
        list(
            qkv = lin(paste0(p, "attn.to_qkv_mlp_proj.weight")),
            out = lin(paste0(p, "attn.to_out.weight")),
            norm_q = vec(paste0(p, "attn.norm_q.weight")),
            norm_k = vec(paste0(p, "attn.norm_k.weight"))
        )
    })

    w
}

#' Load Qwen3-4B text-encoder weights for FLUX.2 into an anvl pytree
#'
#' Reads the sharded \code{text_encoder} checkpoint (bf16 upcast to f32).
#' The embedding table stays an R matrix for host-side gather
#' (\code{\link{yq_qwen3_embed}}); only the first \code{n_layers} decoder
#' layers are loaded (klein consumes mid-stack states, so the deeper
#' layers and the tied LM head are never needed). Each tensor is wrapped
#' as an \code{AnvlArray} as it is read, freeing the R copy.
#'
#' @param dir The \code{text_encoder} directory (index + shards).
#' @param n_layers Integer. Decoder layers to load (klein: 27, enough
#'   for out_layers up to 27).
#' @param device Character. Target device.
#'
#' @return List \code{list(embed = <R matrix [vocab, hidden]>,
#'   layers = <list of per-layer weight lists>)}.
#'
#' @export
yq_qwen3_load_weights <- function(dir, n_layers = 27L, device = "cpu") {
    st <- .yq_st_open_sharded(dir)
    on.exit(.yq_st_close(st))
    lin <- function(key) anvl::nv_array(.yq_st_read(st, key, transpose = TRUE),
                                        dtype = "f32", device = device)
    vec <- function(key) anvl::nv_array(.yq_st_read(st, key),
                                        dtype = "f32", device = device)

    embed <- .yq_st_read(st, "model.embed_tokens.weight")   # [vocab, hidden]

    layers <- lapply(seq_len(n_layers) - 1L, function(i) {
        p <- sprintf("model.layers.%d.", i)
        list(
            in_ln = vec(paste0(p, "input_layernorm.weight")),
            post_ln = vec(paste0(p, "post_attention_layernorm.weight")),
            q_proj = lin(paste0(p, "self_attn.q_proj.weight")),
            k_proj = lin(paste0(p, "self_attn.k_proj.weight")),
            v_proj = lin(paste0(p, "self_attn.v_proj.weight")),
            o_proj = lin(paste0(p, "self_attn.o_proj.weight")),
            q_norm = vec(paste0(p, "self_attn.q_norm.weight")),
            k_norm = vec(paste0(p, "self_attn.k_norm.weight")),
            gate = lin(paste0(p, "mlp.gate_proj.weight")),
            up = lin(paste0(p, "mlp.up_proj.weight")),
            down = lin(paste0(p, "mlp.down_proj.weight"))
        )
    })

    list(embed = embed, layers = layers)
}
