% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_gelu}
\alias{yq_gelu}
\title{GELU activation}
\usage{
yq_gelu(x, approximate = c("tanh", "none"))
}
\arguments{
\item{x}{AnvlArray.}

\item{approximate}{Character. \code{"tanh"} (default) or \code{"none"}.}
}
\description{
Gaussian Error Linear Unit. The \code{"tanh"} approximation
(transformers' \code{gelu_new}) is what T5, Gemma, FLUX feed-forwards,
and most GELU transformer ports actually use; \code{"none"} is the
exact erf form.
}
