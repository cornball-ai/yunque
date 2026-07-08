% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_rope_split}
\alias{yq_rope_split}
\title{Apply split-half rotary embeddings (Llama/Qwen convention)}
\usage{
yq_rope_split(x, cos, sin)
}
\arguments{
\item{x}{AnvlArray \code{[B, H, S, D]}, D even.}

\item{cos}{AnvlArray broadcastable to \code{[B, H, S, D/2]}.}

\item{sin}{AnvlArray broadcastable to \code{[B, H, S, D/2]}.}
}
\description{
Rotates pairs formed by splitting the last dimension in half: element
i pairs with element i + D/2 (\code{rotate_half}). Distinct from the
interleaved-pair \code{\link{yq_rope_apply}} used by FLUX.
}
