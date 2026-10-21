% tinyrox says don't edit this manually, but it can't stop you!
\name{causal_mask}
\alias{causal_mask}
\title{Additive causal mask}
\usage{
causal_mask(len, value = -1e+09, dtype = "f32", device = "cpu")
}
\arguments{
\item{len}{Integer. Sequence length \code{S}.}

\item{value}{Numeric. Bias at masked positions.}

\item{dtype}{Character.}

\item{device}{Character.}
}
\value{
AnvlArray \code{[1, 1, S, S]}.
}
\description{
A \code{[1, 1, S, S]} bias with \code{value} above the diagonal, so
query \code{i} attends keys \code{1..i} only. It broadcasts over batch
and heads in \code{\link{sdpa}}; add a \code{\link{seq_mask}} to it with
\code{\link{badd}} for padded causal batches.
}
