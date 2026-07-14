% tinyrox says don't edit this manually, but it can't stop you!
\name{rel_position_attention}
\alias{rel_position_attention}
\title{Relative-position multi-head attention (ESPnet conformer)}
\usage{
rel_position_attention(q, k, v, p, pos_bias_u, pos_bias_v, mask = NULL)
}
\arguments{
\item{q}{AnvlArray \code{[B, H, T, d_k]}. Projected queries.}

\item{k}{AnvlArray \code{[B, H, T, d_k]}. Projected keys.}

\item{v}{AnvlArray \code{[B, H, T, d_k]}. Projected values.}

\item{p}{AnvlArray \code{[B, H, 2T-1, d_k]} (or batch 1, broadcast).
Projected relative-position keys.}

\item{pos_bias_u}{AnvlArray \code{[H, d_k]}. Content bias (\code{u}).}

\item{pos_bias_v}{AnvlArray \code{[H, d_k]}. Position bias (\code{v}).}

\item{mask}{AnvlArray additive bias broadcast to \code{[B, H, T, T]}, or
NULL. Masked positions carry a large negative value (e.g. -65504 for
f16), matching \code{\link{sdpa}}'s additive-mask convention.}
}
\value{
AnvlArray \code{[B, H, T, d_k]}.
}
\description{
Transformer-XL-style relative-position attention: the content term
\code{(q + u) k^T} plus the position term \code{rel_shift((q + v) p^T)},
scaled and softmaxed. Takes already-projected \code{q}/\code{k}/\code{v}
and the projected relative-position keys \code{p} (the caller does the
\code{linear_q/k/v/pos} projections and the output projection, as with
\code{\link{sdpa}}). Returns the attention output before the output
projection.
}
