% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_mish}
\alias{yq_mish}
\title{Mish activation}
\usage{
yq_mish(x)
}
\arguments{
\item{x}{AnvlArray.}
}
\description{
\code{x * tanh(softplus(x))} (Misra 2019); used by chatterbox's CFM
decoder.
}
