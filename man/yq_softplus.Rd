% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_softplus}
\alias{yq_softplus}
\title{Softplus activation}
\usage{
yq_softplus(x)
}
\arguments{
\item{x}{AnvlArray.}
}
\description{
\code{log(1 + exp(x))}, computed in the numerically stable form
\code{max(x, 0) + log1p(exp(-|x|))} (matches \code{torch::nnf_softplus}
with the default \code{beta = 1}).
}
