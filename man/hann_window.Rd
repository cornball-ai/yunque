% tinyrox says don't edit this manually, but it can't stop you!
\name{hann_window}
\alias{hann_window}
\title{Periodic Hann window}
\usage{
hann_window(n)
}
\arguments{
\item{n}{Integer. Window length.}
}
\value{
Numeric vector of length \code{n}.
}
\description{
The \code{periodic = TRUE} window torch/librosa use for spectral
analysis (denominator \code{n}, not \code{n - 1}).
}
