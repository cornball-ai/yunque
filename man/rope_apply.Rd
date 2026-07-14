% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_apply}
\alias{rope_apply}
\title{Apply rotary position embeddings (interleaved pairs)}
\usage{
rope_apply(x, cos, sin)
}
\arguments{
\item{x}{AnvlArray \code{[B, H, S, D]}, D even.}

\item{cos}{AnvlArray with the same shape as \code{x}.}

\item{sin}{AnvlArray with the same shape as \code{x}.}
}
\description{
FLUX convention: the last dimension holds \code{D/2} interleaved
(real, imag) pairs; \code{cos}/\code{sin} carry each angle twice so
they multiply elementwise. All three arguments must share the same
4-D shape \code{[B, H, S, D]} — broadcast \code{cos}/\code{sin}
before calling.
}
