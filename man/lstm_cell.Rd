% tinyrox says don't edit this manually, but it can't stop you!
\name{lstm_cell}
\alias{lstm_cell}
\title{One LSTM timestep}
\usage{
lstm_cell(x_t, h, c, w_ih, w_hh, b_ih = NULL, b_hh = NULL)
}
\arguments{
\item{x_t}{AnvlArray \code{[B, input]}.}

\item{h}{AnvlArray \code{[B, hidden]}. Previous hidden state.}

\item{c}{AnvlArray \code{[B, hidden]}. Previous cell state.}

\item{w_ih}{AnvlArray \code{[input, 4 * hidden]}.}

\item{w_hh}{AnvlArray \code{[hidden, 4 * hidden]}.}

\item{b_ih}{AnvlArray \code{[4 * hidden]} or NULL.}

\item{b_hh}{AnvlArray \code{[4 * hidden]} or NULL.}
}
\value{
List with \code{h} and \code{c}, each \code{[B, hidden]}.
}
\description{
One LSTM timestep
}
