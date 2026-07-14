% tinyrox says don't edit this manually, but it can't stop you!
\name{istft}
\alias{istft}
\title{Inverse short-time Fourier transform}
\usage{
istft(real, imag, n_fft, hop_length = n_fft\%/\%4L, window = NULL, center = TRUE,
      length = NULL)
}
\arguments{
\item{real}{AnvlArray \code{[batch, n_freqs, n_frames]}. Real part.}

\item{imag}{AnvlArray \code{[batch, n_freqs, n_frames]}. Imag part.}

\item{n_fft}{Integer. FFT / window size.}

\item{hop_length}{Integer. Stride between frames.}

\item{window}{Numeric vector of length \code{n_fft}, or NULL for
rectangular.}

\item{center}{Logical. Whether the forward transform used centered
framing (trims \code{n_fft / 2} from each end).}

\item{length}{Integer or NULL. Trim/limit the output to this many
samples.}
}
\value{
AnvlArray \code{[batch, samples]}.
}
\description{
Overlap-add reconstruction (the adjoint of \code{\link{stft}} via
\code{\link{conv_transpose1d}}) with window-sum-square normalization.
With the same \code{window} used for the forward transform and a COLA
hop, recovers the original signal.
}
