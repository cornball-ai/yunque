% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_elu}
\alias{yq_elu}
\title{ELU activation}
\usage{
yq_elu(x, alpha = 1)
}
\arguments{
\item{x}{AnvlArray.}

\item{alpha}{Numeric. Negative-saturation scale (default 1).}
}
\description{
\code{x} for \code{x > 0}, else \code{alpha * (exp(x) - 1)}. Written
branch-free as \code{max(x, 0) + alpha * (exp(min(x, 0)) - 1)} to avoid a
predicated select. Matches \code{torch::nnf_elu}.
}
