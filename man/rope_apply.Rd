% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_apply}
\alias{rope_apply}
\title{Apply rotary position embeddings (interleaved pairs)}
\usage{
rope_apply(x, cos, sin)
}
\arguments{
\item{x}{AnvlArray \code{[B, H, S, D]}, D even.}

\item{cos}{AnvlArray \code{[B, H, S, D]}, \code{[B, S, D]} or
\code{[S, D]}.}

\item{sin}{AnvlArray, same shapes as \code{cos}.}
}
\description{
FLUX convention: the last dimension holds \code{D/2} interleaved
(real, imag) pairs; \code{cos}/\code{sin} carry each angle twice so
they multiply elementwise. The tables may be \code{[B, H, S, D]} like
\code{x}, a \code{[S, D]} table shared across batch and heads, or a
per-batch \code{[B, S, D]} table, which gets the head axis inserted.
}
