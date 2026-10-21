#' Adaptive layer-norm modulation
#'
#' \code{x * (1 + scale) + shift}, the adaLN modulation DiT-family blocks
#' (FLUX, LTX, Z-Image, SD3) apply after a parameter-free layer norm.
#' \code{shift} and \code{scale} are per-batch vectors \code{[B, D]} from
#' the timestep or conditioning embedding; they are aligned to
#' \code{x [B, S, D]} by inserting the sequence axis, so the caller does
#' no reshape or broadcast. Already aligned \code{[B, 1, D]} or
#' \code{[B, S, D]} inputs work too.
#'
#' @param x AnvlArray \code{[B, S, D]}, typically a normalized activation.
#' @param shift AnvlArray \code{[B, D]}, or broadcastable to \code{x}.
#' @param scale AnvlArray \code{[B, D]}, or broadcastable to \code{x}.
#'
#' @return AnvlArray with the shape of \code{x}.
#'
#' @export
modulate <- function(x, shift, scale) {
    x * (.align_batched(scale, x) + 1) + .align_batched(shift, x)
}

#' Gated residual connection
#'
#' \code{h + gate * y}: adds a branch output \code{y} to the residual
#' stream \code{h}, scaled by a per-batch gate \code{[B, D]} from the
#' conditioning embedding (the third chunk of an adaLN modulation). The
#' gate is aligned to \code{h} the way \code{\link{modulate}} aligns its
#' arguments.
#'
#' @param h AnvlArray \code{[B, S, D]}. Residual stream.
#' @param gate AnvlArray \code{[B, D]}, or broadcastable to \code{h}.
#' @param y AnvlArray with the shape of \code{h}. Branch output.
#'
#' @return AnvlArray with the shape of \code{h}.
#'
#' @export
gated_residual <- function(h, gate, y) {
    h + .align_batched(gate, h) * y
}

#' Sinusoidal timestep embedding
#'
#' The diffusers \code{get_timestep_embedding()}: with
#' \code{half = dim \%/\% 2}, frequencies
#' \code{exp(-log(max_period) * i / (half - downscale_freq_shift))} for
#' \code{i} in \code{0:(half - 1)}, then \code{[sin(t f), cos(t f)]} per
#' timestep, or cos first with \code{flip_sin_to_cos}. FLUX and SDXL use
#' \code{flip_sin_to_cos = TRUE, downscale_freq_shift = 0}; the DDPM
#' default is \code{FALSE, 1}. An odd \code{dim} is zero-padded by one
#' column. The frequencies are a constant at trace time, so inside
#' \code{anvl::jit()} only the outer product and the trig run on device.
#'
#' @param timesteps AnvlArray \code{[B]}, or a numeric vector.
#' @param dim Integer. Embedding width.
#' @param max_period Numeric. Longest period (default 10000).
#' @param flip_sin_to_cos Logical. Put cos before sin.
#' @param downscale_freq_shift Numeric. Subtracted from \code{half} in the
#'   frequency denominator.
#' @param scale Numeric. Multiplies the angles before sin and cos.
#' @param dtype Character. Dtype used when \code{timesteps} is a plain
#'   numeric vector; an AnvlArray keeps its own.
#'
#' @return AnvlArray \code{[B, dim]}.
#'
#' @export
timestep_embedding <- function(timesteps, dim, max_period = 10000,
                               flip_sin_to_cos = FALSE,
                               downscale_freq_shift = 1, scale = 1,
                               dtype = "f32") {
    if (anvl::is_arrayish(timesteps)) {
        dtype <- as.character(anvl::dtype(timesteps))
    } else {
        timesteps <- anvl::nv_array(as.numeric(timesteps), dtype = dtype)
    }
    half <- dim %/% 2L
    freqs <- exp(-log(max_period) * (seq_len(half) - 1) /
                 (half - downscale_freq_shift))
    angles <- anvl::nv_outer(timesteps, anvl::nv_array(freqs, dtype = dtype))
    if (scale != 1) {
        angles <- angles * scale
    }
    sn <- anvl::nv_sin(angles)
    cs <- anvl::nv_cos(angles)
    if (flip_sin_to_cos) {
        emb <- anvl::nv_concatenate(cs, sn, axis = 2L)
    } else {
        emb <- anvl::nv_concatenate(sn, cs, axis = 2L)
    }
    if (dim %% 2L == 1L) {
        emb <- anvl::nv_pad(emb, 0, c(0L, 0L), c(0L, 1L))
    }
    emb
}
