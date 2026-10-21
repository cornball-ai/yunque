% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_split}
\alias{rope_split}
\title{Apply split-half rotary embeddings (Llama/Qwen convention)}
\usage{
rope_split(x, cos, sin)
}
\arguments{
\item{x}{AnvlArray \code{[B, H, S, D]}, D even.}

\item{cos}{AnvlArray \code{[B, H, S, D/2]}, or a lower-rank table that
aligns to it: \code{[S, D/2]} shared across batch and heads, or a
per-batch \code{[B, S, D/2]}, which gets the head axis inserted.}

\item{sin}{AnvlArray, same shapes as \code{cos}.}
}
\description{
Rotates pairs formed by splitting the last dimension in half: element
i pairs with element i + D/2 (\code{rotate_half}). Distinct from the
interleaved-pair \code{\link{rope_apply}} used by FLUX.
}
