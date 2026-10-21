% tinyrox says don't edit this manually, but it can't stop you!
\name{broadcasting}
\alias{broadcasting}
\title{Broadcasting arithmetic}
\arguments{
\item{x}{AnvlArray, or a single number.}

\item{y}{AnvlArray, or a single number.}
}
\value{
AnvlArray of the broadcast shape.
}
\description{
anvl's arithmetic operators broadcast scalars only: two arrays of
different shapes are an error, even \code{[B, S, D]} against
\code{[D]}. These helpers apply numpy-style broadcasting first (ranks
align on the right, size-1 axes expand), then run the op. anvl's own
\code{nv_broadcast_arrays()} aligns ranks on the left, like base R, so
it would pair \code{[D]} with \code{B}. Anything numpy would refuse is
an error here too, so a shape mismatch is caught at trace time rather
than guessed at. A plain R number is passed straight to the operator.
}
