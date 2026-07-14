% tinyrox says don't edit this manually, but it can't stop you!
\name{batch_norm}
\alias{batch_norm}
\title{Batch normalization (inference) over the channel dimension}
\usage{
batch_norm(x, running_mean, running_var, weight = NULL, bias = NULL, eps = 1e-05)
}
\arguments{
\item{x}{AnvlArray \code{[N, C, ...]}.}

\item{running_mean}{AnvlArray \code{[C]}. Per-channel running mean.}

\item{running_var}{AnvlArray \code{[C]}. Per-channel running variance.}

\item{weight}{AnvlArray \code{[C]} scale, or NULL.}

\item{bias}{AnvlArray \code{[C]} shift, or NULL.}

\item{eps}{Numeric. Stability epsilon.}
}
\description{
Normalizes each channel by its stored running statistics, then applies
an optional per-channel affine: \code{(x - mean) / sqrt(var + eps) *
weight + bias}. Matches \code{torch::nnf_batch_norm} in eval mode.
Channel is dim 2; works for any \code{[N, C, ...]} rank.
}
