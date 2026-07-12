% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdpa}
\alias{yq_sdpa}
\title{Scaled dot-product attention}
\usage{
yq_sdpa(query, key, value, mask = NULL, scale = NULL, precision = "highest")
}
\arguments{
\item{query}{AnvlArray \code{[B, H, S, D]}.}

\item{key}{AnvlArray \code{[B, H, S, D]}.}

\item{value}{AnvlArray \code{[B, H, S, D]}.}

\item{mask}{AnvlArray additive bias added to the (scaled) scores,
broadcast to the score shape \code{[B, H, Sq, Sk]} — a causal /
padding mask, or a learned bias (e.g. T5 relative-position bias).
NULL for none.}

\item{scale}{Numeric score scale. NULL (default) uses
\code{1 / sqrt(head_dim)}; pass \code{1} for the unscaled attention
T5 uses (its relative-position bias is added to raw \code{qk^T}).}

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
