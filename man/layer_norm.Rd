% tinyrox says don't edit this manually, but it can't stop you!
\name{layer_norm}
\alias{layer_norm}
\title{Layer normalization over the last dimension}
\usage{
layer_norm(x, weight = NULL, bias = NULL, eps = 1e-06)
}
\arguments{
\item{x}{AnvlArray.}

\item{weight}{AnvlArray of length \code{shape(x)[ndims(x)]}, or NULL for no
scale.}

\item{bias}{AnvlArray of length \code{shape(x)[ndims(x)]}, or NULL for no
shift.}

\item{eps}{Numeric. Stability epsilon.}
}
\description{
Biased variance, matching \code{torch::nn_layer_norm}. With
\code{weight} and \code{bias} both NULL it is the affine-free form
(\code{elementwise_affine = FALSE}); pass a length-\code{shape(x)[ndims(x)]}
\code{weight} (and optional \code{bias}) for the standard affine LayerNorm
\code{normalized * weight + bias}.
}
