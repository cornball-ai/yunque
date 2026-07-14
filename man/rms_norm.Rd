% tinyrox says don't edit this manually, but it can't stop you!
\name{rms_norm}
\alias{rms_norm}
\title{RMS normalization over the last dimension}
\usage{
rms_norm(x, weight = NULL, eps = 1e-06)
}
\arguments{
\item{x}{AnvlArray.}

\item{weight}{AnvlArray of length \code{shape(x)[ndims(x)]}, or NULL.}

\item{eps}{Numeric. Stability epsilon.}
}
\description{
Matches \code{diffuseR::ltx23_rms_norm} (float32 compute is native
here).
}
