% tinyrox says don't edit this manually, but it can't stop you!
\name{group_norm}
\alias{group_norm}
\title{Group normalization over [B, C, H, W]}
\usage{
group_norm(x, weight, bias, num_groups = 32L, eps = 1e-06)
}
\arguments{
\item{x}{AnvlArray \code{[B, C, H, W]}.}

\item{weight}{AnvlArray \code{[C]} scale.}

\item{bias}{AnvlArray \code{[C]} shift.}

\item{num_groups}{Integer.}

\item{eps}{Numeric.}
}
\description{
Normalizes within \code{num_groups} channel groups (over the group's
channels and all spatial positions), then applies a per-channel
affine. Matches \code{torch::nn_group_norm}.
}
