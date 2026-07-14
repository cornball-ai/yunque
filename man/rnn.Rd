% tinyrox says don't edit this manually, but it can't stop you!
\name{rnn}
\alias{rnn}
\title{Recurrent layers for anvl}
\description{
A torch-compatible LSTM built from anvl primitives. The recurrence is
unrolled over time at trace time (fine for the bounded sequences these
ports use, e.g. speaker-embedding partials); a rolled \code{nv_while}
version is a future optimization for long sequences. Weights use the
\code{\link{linear}} layout: \code{w_ih} is \code{[input, 4 * hidden]}
and \code{w_hh} is \code{[hidden, 4 * hidden]} (the transpose of torch's
\code{[4 * hidden, .]} \code{weight_ih_l*} / \code{weight_hh_l*}, which
\code{\link{st_read}} produces for free). Gate order is torch's:
input, forget, cell, output.
}
