% tinyrox says don't edit this manually, but it can't stop you!
\name{w_linear}
\alias{w_linear}
\title{Linear weight spec}
\usage{
w_linear(key)
}
\arguments{
\item{key}{Character. Tensor name in the checkpoint.}
}
\value{
A weight spec.
}
\description{
Describes a \code{[out, in]} checkpoint matrix to be returned in the
\code{[in, out]} layout \code{\link{linear}} takes. See
\code{\link{load_weights}}.
}
