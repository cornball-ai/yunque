% tinyrox says don't edit this manually, but it can't stop you!
\name{pad_to}
\alias{pad_to}
\title{Pad an axis up to a multiple}
\usage{
pad_to(x, multiple, dim = 2L, value = 0)
}
\arguments{
\item{x}{AnvlArray.}

\item{multiple}{Integer. Bucket size.}

\item{dim}{Integer. Axis to pad; default 2, the sequence axis of
\code{[B, S, ...]}.}

\item{value}{Numeric. Fill value.}
}
\value{
AnvlArray, or \code{x} itself when its length is already a
  multiple.
}
\description{
XLA compiles one executable per input shape, so a model fed variable
sequence lengths recompiles at every new length. Padding the sequence
axis up to the next multiple of \code{multiple} caps the distinct
shapes, and compilations, at one per bucket. Pair it with
\code{\link{seq_mask}} so padded positions neither attend nor get
attended to. The padded length is \code{shape(out)[dim]}.
}
