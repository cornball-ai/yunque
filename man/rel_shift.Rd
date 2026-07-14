% tinyrox says don't edit this manually, but it can't stop you!
\name{rel_shift}
\alias{rel_shift}
\title{Relative-position shift (Transformer-XL / ESPnet)}
\usage{
rel_shift(x)
}
\arguments{
\item{x}{AnvlArray \code{[B, H, T, 2T-1]}.}
}
\value{
AnvlArray \code{[B, H, T, T]}.
}
\description{
Turns the \code{[B, H, T, 2T-1]} query-vs-relative-position scores into
\code{[B, H, T, T]} query-vs-key scores via the pad / reshape / slice
trick, so entry \code{[i, t]} picks the score at relative distance
\code{i - t}. Relies on \code{nv_reshape} being row-major (it is).
}
