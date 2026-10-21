% tinyrox says don't edit this manually, but it can't stop you!
\name{w_tensor}
\alias{w_tensor}
\title{Tensor weight spec}
\usage{
w_tensor(key, transpose = FALSE)
}
\arguments{
\item{key}{Character. Tensor name in the checkpoint.}

\item{transpose}{Logical. Transpose a 2-D tensor on read.}
}
\value{
A weight spec.
}
\description{
Describes a tensor to be returned as stored: norm weights, biases,
conv kernels, embedding tables. See \code{\link{load_weights}}.
}
