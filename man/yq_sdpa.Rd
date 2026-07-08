% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdpa}
\alias{yq_sdpa}
\title{Scaled dot-product attention}
\usage{
yq_sdpa(query, key, value, precision = "highest")
}
\arguments{
\item{query}{AnvlArray \code{[B, H, S, D]}.}

\item{key}{AnvlArray \code{[B, H, S, D]}.}

\item{value}{AnvlArray \code{[B, H, S, D]}.}

\item{precision}{Character. Matmul precision; the \code{"highest"}
default forbids TF32-style downgrades so f32 is honest f32.
Ignored on anvl 0.3.0, whose \code{nv_matmul} has no precision
parameter (and whose CUDA dots run TF32 regardless).}
}
\value{
AnvlArray \code{[B, H, Sq, D]}.
}
\description{
Composed from batched matmul and softmax; the full
\code{[B, H, Sq, Sk]} score matrix materializes (XLA may fuse, but
budget for the worst case at long sequence lengths).
}
