% tinyrox says don't edit this manually, but it can't stop you!
\name{modulate}
\alias{modulate}
\title{Adaptive layer-norm modulation}
\usage{
modulate(x, shift, scale)
}
\arguments{
\item{x}{AnvlArray \code{[B, S, D]}, typically a normalized activation.}

\item{shift}{AnvlArray \code{[B, D]}, or broadcastable to \code{x}.}

\item{scale}{AnvlArray \code{[B, D]}, or broadcastable to \code{x}.}
}
\value{
AnvlArray with the shape of \code{x}.
}
\description{
\code{x * (1 + scale) + shift}, the adaLN modulation DiT-family blocks
(FLUX, LTX, Z-Image, SD3) apply after a parameter-free layer norm.
\code{shift} and \code{scale} are per-batch vectors \code{[B, D]} from
the timestep or conditioning embedding; they are aligned to
\code{x [B, S, D]} by inserting the sequence axis, so the caller does
no reshape or broadcast. Already aligned \code{[B, 1, D]} or
\code{[B, S, D]} inputs work too.
}
