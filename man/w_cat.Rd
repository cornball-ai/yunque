% tinyrox says don't edit this manually, but it can't stop you!
\name{w_cat}
\alias{w_cat}
\title{Concatenated weight spec}
\usage{
w_cat(keys, dim, transpose = TRUE)
}
\arguments{
\item{keys}{Character vector of tensor names.}

\item{dim}{Integer. Axis to concatenate along, in the returned layout:
1 or 2 for matrices, ignored for vectors.}

\item{transpose}{Logical. Transpose 2-D tensors on read.}
}
\value{
A weight spec.
}
\description{
Describes several checkpoint tensors concatenated along \code{dim}
after reading, for a fused projection such as one QKV weight assembled
from separate q, k and v matrices. Matrices are read transposed as
\code{\link{w_linear}} does, so \code{dim = 2} stacks their output
columns; pass \code{transpose = FALSE} for vectors (biases) joined end
to end. See \code{\link{load_weights}}.
}
