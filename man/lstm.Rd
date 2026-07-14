% tinyrox says don't edit this manually, but it can't stop you!
\name{lstm}
\alias{lstm}
\title{Multi-layer LSTM forward pass}
\usage{
lstm(x, layers, batch_first = FALSE)
}
\arguments{
\item{x}{AnvlArray. \code{[seq, batch, input]}, or
\code{[batch, seq, input]} when \code{batch_first = TRUE}.}

\item{layers}{List of per-layer parameter lists, each with elements
\code{w_ih}, \code{w_hh}, \code{b_ih}, \code{b_hh} (biases may be
NULL). Layer 1's \code{input} is \code{x}'s feature size; deeper
layers' is the previous \code{hidden}.}

\item{batch_first}{Logical. Layout of \code{x} and the returned
\code{output}.}
}
\value{
List with \code{output} (all top-layer hidden states, same
  layout as \code{x}), \code{h_n} and \code{c_n}
  (\code{[num_layers, batch, hidden]}, each layer's final state).
}
\description{
Stacked unidirectional LSTM matching \code{torch::nn_lstm} with
\code{bidirectional = FALSE}. Each layer feeds its full output sequence
to the next.
}
