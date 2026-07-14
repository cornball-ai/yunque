% tinyrox says don't edit this manually, but it can't stop you!
\name{snake}
\alias{snake}
\title{Snake activation}
\usage{
snake(x, alpha, beta = NULL, eps = 1e-09)
}
\arguments{
\item{x}{AnvlArray.}

\item{alpha}{AnvlArray. Frequency parameter, broadcastable to \code{x}.}

\item{beta}{AnvlArray or NULL. Magnitude parameter; NULL reuses
\code{alpha}.}

\item{eps}{Numeric. Guards the reciprocal against division by zero.}
}
\description{
\code{x + sin(alpha * x)^2 / (beta + eps)} (Ziyin et al. 2020). The
single-parameter Snake HiFiGAN uses sets \code{beta = alpha}; the
SnakeBeta variant (BigVGAN / LTX vocoders) passes an independent
\code{beta}. \code{alpha}/\code{beta} must be broadcastable to \code{x}
(same rank, size-1 dims expand) -- reshape a per-channel \code{[C]}
parameter to \code{[1, C, 1]} before calling.
}
