% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_slice_seq}
\alias{yq_slice_seq}
\title{Slice a contiguous range along the sequence dimension (dim 2)}
\usage{
yq_slice_seq(x, from, to)
}
\arguments{
\item{x}{AnvlArray \code{[B, S, ...]}.}

\item{from}{Integer. 1-based inclusive start.}

\item{to}{Integer. 1-based inclusive end.}
}
\description{
For \code{[B, S, D]} tensors: split text/image token spans of a
concatenated sequence.
}
