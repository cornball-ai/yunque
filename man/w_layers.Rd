% tinyrox says don't edit this manually, but it can't stop you!
\name{w_layers}
\alias{w_layers}
\title{Per-layer weight specs}
\usage{
w_layers(n, spec)
}
\arguments{
\item{n}{Integer. Number of layers.}

\item{spec}{Function of the 1-based layer index returning a spec tree.}
}
\value{
A list of \code{n} spec trees.
}
\description{
Evaluates \code{spec(i)} for \code{i} in \code{1:n}, giving the list of
per-layer spec trees a stack of identical blocks needs. The function
builds the checkpoint keys from the index (checkpoints count from 0).
See \code{\link{load_weights}}.
}
