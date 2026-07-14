% tinyrox says don't edit this manually, but it can't stop you!
\name{stft}
\alias{stft}
\title{Short-time Fourier transform}
\usage{
stft(signal, n_fft, hop_length = n_fft\%/\%4L, window = NULL, center = TRUE)
}
\arguments{
\item{signal}{AnvlArray \code{[batch, samples]}.}

\item{n_fft}{Integer. FFT / window size.}

\item{hop_length}{Integer. Stride between frames.}

\item{window}{Numeric vector of length \code{n_fft}, or NULL for a
rectangular (all-ones) window.}

\item{center}{Logical. If TRUE, reflect-pad the signal by
\code{n_fft / 2} each side so frame \code{t} is centered at
\code{t * hop_length} (torch default).}
}
\value{
List with \code{real} and \code{imag}, each AnvlArray
  \code{[batch, n_freqs, n_frames]} where \code{n_freqs = n_fft / 2 + 1}.
}
\description{
Short-time Fourier transform
}
