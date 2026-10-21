% tinyrox says don't edit this manually, but it can't stop you!
\name{seq_mask}
\alias{seq_mask}
\title{Additive padding mask from sequence lengths}
\usage{
seq_mask(lengths, len, value = -1e+09, dtype = "f32", device = "cpu")
}
\arguments{
\item{lengths}{Integer vector \code{[B]} of valid lengths.}

\item{len}{Integer. Padded sequence length \code{S}.}

\item{value}{Numeric. Bias at padded positions. The default suits f32;
use \code{-65504} for f16.}

\item{dtype}{Character.}

\item{device}{Character.}
}
\value{
AnvlArray \code{[B, 1, 1, S]}.
}
\description{
Builds the \code{[B, 1, 1, S]} additive bias \code{\link{sdpa}} takes:
0 at valid key positions and \code{value}, a large negative number, at
padded ones, so padded keys get no attention weight. Lengths are
host-side integers, so the mask is built in R and uploaded once; inside
\code{anvl::jit()} pass it in as an input.
}
