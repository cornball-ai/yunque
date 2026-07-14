#' Short-time Fourier transform for anvl
#'
#' anvl has no FFT, so the STFT is a windowed DFT expressed as a
#' convolution against a precomputed cos/-sin Fourier basis (stride =
#' hop), and the inverse is the adjoint via \code{\link{conv_transpose1d}}
#' plus window-overlap normalization. Matches \code{torch::torch_stft} /
#' \code{torch::torch_istft} conventions (one-sided, periodic window,
#' \code{center} reflect padding). The frontend runs once per clip, off the
#' hot path, so the full \code{[n_freqs, n_fft]} basis materializing is
#' fine.
#'
#' @name stft-frontend
NULL

#' Periodic Hann window
#'
#' The \code{periodic = TRUE} window torch/librosa use for spectral
#' analysis (denominator \code{n}, not \code{n - 1}).
#'
#' @param n Integer. Window length.
#'
#' @return Numeric vector of length \code{n}.
#'
#' @export
hann_window <- function(n) {
    0.5 - 0.5 * cos(2 * pi * (0:(n - 1L)) / n)
}

# Forward DFT basis [n_freqs * 2, 1, n_fft]: cos rows then -sin rows, each
# times the window. Laid out as a conv1d weight (C_out = n_freqs * 2).
.stft_basis <- function(n_fft, win) {
    nf <- n_fft %/% 2L + 1L
    n <- 0:(n_fft - 1L)
    basis <- matrix(0, nf * 2L, n_fft)
    for (k in 0:(nf - 1L)) {
        basis[k + 1L, ] <- cos(2 * pi * k * n / n_fft) * win
        basis[nf + k + 1L, ] <- -sin(2 * pi * k * n / n_fft) * win
    }
    array(basis, c(nf * 2L, 1L, n_fft))
}

# Reflect-pad the last dim of [B, 1, L] by `pad` each side (torch 'reflect':
# mirror without repeating the edge sample). anvl's nv_pad is zero-only.
.reflect_pad_1d <- function(x, pad) {
    len <- anvl::shape(x)[3L]
    left <- anvl::nv_reverse(.slice_dim(x, 3L, 2L, pad + 1L), dims = 3L)
    right <- anvl::nv_reverse(.slice_dim(x, 3L, len - pad, len - 1L), dims = 3L)
    anvl::nv_concatenate(left, x, right, dimension = 3L)
}

#' Short-time Fourier transform
#'
#' @param signal AnvlArray \code{[batch, samples]}.
#' @param n_fft Integer. FFT / window size.
#' @param hop_length Integer. Stride between frames.
#' @param window Numeric vector of length \code{n_fft}, or NULL for a
#'   rectangular (all-ones) window.
#' @param center Logical. If TRUE, reflect-pad the signal by
#'   \code{n_fft / 2} each side so frame \code{t} is centered at
#'   \code{t * hop_length} (torch default).
#'
#' @return List with \code{real} and \code{imag}, each AnvlArray
#'   \code{[batch, n_freqs, n_frames]} where \code{n_freqs = n_fft / 2 + 1}.
#'
#' @export
stft <- function(signal, n_fft, hop_length = n_fft %/% 4L, window = NULL,
                 center = TRUE) {
    s <- anvl::shape(signal)
    batch <- s[1L]
    win <- if (is.null(window)) rep(1, n_fft) else window
    basis_r <- .stft_basis(n_fft, win)
    basis <- anvl::nv_array(c(basis_r), dtype = anvl::dtype(signal),
                            shape = dim(basis_r))
    x <- anvl::nv_reshape(signal, c(batch, 1L, s[2L]))
    if (center) {
        x <- .reflect_pad_1d(x, n_fft %/% 2L)
    }
    spec <- conv1d(x, basis, stride = as.integer(hop_length))
    nf <- n_fft %/% 2L + 1L
    list(real = .slice_dim(spec, 2L, 1L, nf),
         imag = .slice_dim(spec, 2L, nf + 1L, 2L * nf))
}

# Inverse DFT basis [n_freqs * 2, 1, n_fft] as a conv_transpose1d weight
# (C_in = n_freqs * 2 spectrum channels, C_out = 1). a_k folds the
# one-sided -> full spectrum symmetry: a_0 = a_Nyquist = 1, else 2.
.istft_basis <- function(n_fft, win) {
    nf <- n_fft %/% 2L + 1L
    n <- 0:(n_fft - 1L)
    a <- rep(2, nf)
    a[1L] <- 1
    if (n_fft %% 2L == 0L) {
        a[nf] <- 1
    }
    basis <- matrix(0, nf * 2L, n_fft)
    for (k in 0:(nf - 1L)) {
        basis[k + 1L, ] <- (a[k + 1L] / n_fft) * cos(2 * pi * k * n / n_fft) * win
        basis[nf + k + 1L, ] <- -(a[k + 1L] / n_fft) * sin(2 * pi * k * n / n_fft) * win
    }
    array(basis, c(nf * 2L, 1L, n_fft))
}

# Overlap-add normalization envelope: sum of the squared window placed at
# every frame position. Length (frames - 1) * hop + n_fft.
.window_sumsquare <- function(win, n_fft, hop, frames) {
    len <- (frames - 1L) * hop + n_fft
    env <- numeric(len)
    w2 <- win^2
    for (m in 0:(frames - 1L)) {
        idx <- (m * hop + 1L):(m * hop + n_fft)
        env[idx] <- env[idx] + w2
    }
    env
}

#' Inverse short-time Fourier transform
#'
#' Overlap-add reconstruction (the adjoint of \code{\link{stft}} via
#' \code{\link{conv_transpose1d}}) with window-sum-square normalization.
#' With the same \code{window} used for the forward transform and a COLA
#' hop, recovers the original signal.
#'
#' @param real,imag AnvlArray \code{[batch, n_freqs, n_frames]}.
#' @param n_fft Integer. FFT / window size.
#' @param hop_length Integer. Stride between frames.
#' @param window Numeric vector of length \code{n_fft}, or NULL for
#'   rectangular.
#' @param center Logical. Whether the forward transform used centered
#'   framing (trims \code{n_fft / 2} from each end).
#' @param length Integer or NULL. Trim/limit the output to this many
#'   samples.
#'
#' @return AnvlArray \code{[batch, samples]}.
#'
#' @export
istft <- function(real, imag, n_fft, hop_length = n_fft %/% 4L,
                  window = NULL, center = TRUE, length = NULL) {
    s <- anvl::shape(real)
    batch <- s[1L]
    frames <- s[3L]
    win <- if (is.null(window)) rep(1, n_fft) else window
    hop <- as.integer(hop_length)
    spec <- anvl::nv_concatenate(real, imag, dimension = 2L)
    ibasis_r <- .istft_basis(n_fft, win)
    ibasis <- anvl::nv_array(c(ibasis_r), dtype = anvl::dtype(real),
                             shape = dim(ibasis_r))
    y <- conv_transpose1d(spec, ibasis, stride = hop)
    len_full <- (frames - 1L) * hop + n_fft
    env_r <- pmax(.window_sumsquare(win, n_fft, hop, frames), 1e-11)
    env <- anvl::nv_array(env_r, dtype = anvl::dtype(real),
                          shape = c(1L, 1L, len_full))
    y <- y / anvl::nv_broadcast_to(env, anvl::shape(y))
    lo <- if (center) n_fft %/% 2L + 1L else 1L
    hi <- if (center) len_full - n_fft %/% 2L else len_full
    y <- .slice_dim(y, 3L, lo, hi)
    if (!is.null(length)) {
        y <- .slice_dim(y, 3L, 1L, min(as.integer(length), anvl::shape(y)[3L]))
    }
    anvl::nv_reshape(y, c(batch, anvl::shape(y)[3L]))
}
