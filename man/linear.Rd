% tinyrox says don't edit this manually, but it can't stop you!
\name{linear}
\alias{linear}
\title{Linear layer (bias-free)}
\usage{
linear(x, w_t, bias = NULL, precision = "highest")
}
\arguments{
\item{x}{AnvlArray \code{[..., d_in]}.}

\item{w_t}{AnvlArray \code{[d_in, d_out]}.}

\item{bias}{AnvlArray \code{[d_out]} or NULL.}

\item{precision}{Character. Matmul precision; the \code{"highest"}
default forbids TF32-style downgrades so f32 is honest f32.
Ignored on anvl 0.3.0, whose \code{nv_matmul} has no precision
parameter (and whose CUDA dots run TF32 regardless).}
}
\description{
\code{x [..., d_in] \\%*\\% w_t [d_in, d_out]}. Pass the weight
pre-transposed (torch checkpoints store \code{[d_out, d_in]}); do the
transpose once at load time, not per call. Higher-rank inputs are
flattened to 2-D for the matmul and restored after.
}
