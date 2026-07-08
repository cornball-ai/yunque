% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_layer_norm}
\alias{yq_layer_norm}
\title{Layer normalization (no affine) over the last dimension}
\usage{
yq_layer_norm(x, eps = 1e-06)
}
\arguments{
\item{x}{AnvlArray.}

\item{eps}{Numeric. Stability epsilon.}
}
\description{
Biased variance, matching \code{torch::nn_layer_norm} with
\code{elementwise_affine = FALSE}.
}
