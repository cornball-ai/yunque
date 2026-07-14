% tinyrox says don't edit this manually, but it can't stop you!
\name{stft-frontend}
\alias{stft-frontend}
\title{Short-time Fourier transform for anvl}
\description{
anvl has no FFT, so the STFT is a windowed DFT expressed as a
convolution against a precomputed cos/-sin Fourier basis (stride =
hop), and the inverse is the adjoint via \code{\link{conv_transpose1d}}
plus window-overlap normalization. Matches \code{torch::torch_stft} /
\code{torch::torch_istft} conventions (one-sided, periodic window,
\code{center} reflect padding). The frontend runs once per clip, off the
hot path, so the full \code{[n_freqs, n_fft]} basis materializing is
fine.
}
