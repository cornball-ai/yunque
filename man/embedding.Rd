% tinyrox says don't edit this manually, but it can't stop you!
\name{embedding}
\alias{embedding}
\title{Embedding lookup (host-side gather)}
\usage{
embedding(weight, ids, zero_based = TRUE, dtype = "f32", device = "cpu")
}
\arguments{
\item{weight}{R matrix \code{[num_embeddings, dim]} (e.g. from
\code{\link{st_read}}).}

\item{ids}{Integer vector \code{[N]} or matrix \code{[B, S]} of token ids.}

\item{zero_based}{Logical. TRUE (default) treats \code{ids} as 0-based
(torch convention); FALSE as 1-based R indices.}

\item{dtype}{Character. Output dtype.}

\item{device}{Character.}
}
\value{
AnvlArray \code{[N, dim]} for a vector \code{ids}, or
  \code{[B, S, dim]} for a matrix.
}
\description{
Gathers rows of an embedding table for the given token ids. The table
stays an R matrix (never resident on device); only the gathered result
crosses to anvl -- ids are host-side integers at the input boundary, so
this sidesteps an on-device gather entirely (the pattern the anvl model
ports use for token/position embeddings).
}
