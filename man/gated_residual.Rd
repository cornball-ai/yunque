% tinyrox says don't edit this manually, but it can't stop you!
\name{gated_residual}
\alias{gated_residual}
\title{Gated residual connection}
\usage{
gated_residual(h, gate, y)
}
\arguments{
\item{h}{AnvlArray \code{[B, S, D]}. Residual stream.}

\item{gate}{AnvlArray \code{[B, D]}, or broadcastable to \code{h}.}

\item{y}{AnvlArray with the shape of \code{h}. Branch output.}
}
\value{
AnvlArray with the shape of \code{h}.
}
\description{
\code{h + gate * y}: adds a branch output \code{y} to the residual
stream \code{h}, scaled by a per-batch gate \code{[B, D]} from the
conditioning embedding (the third chunk of an adaLN modulation). The
gate is aligned to \code{h} the way \code{\link{modulate}} aligns its
arguments.
}
