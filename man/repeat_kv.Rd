% tinyrox says don't edit this manually, but it can't stop you!
\name{repeat_kv}
\alias{repeat_kv}
\title{Repeat KV heads to match query heads (grouped-query attention)}
\usage{
repeat_kv(x, groups)
}
\arguments{
\item{x}{AnvlArray \code{[B, KV, S, D]}.}

\item{groups}{Integer. Query heads per KV head.}
}
\value{
AnvlArray \code{[B, KV * groups, S, D]}.
}
\description{
Interleaved expansion (\code{repeat_interleave} over the head dim):
each KV head is repeated \code{groups} times consecutively, so head
\code{j} of the query maps to KV head \code{floor(j / groups)}.
}
