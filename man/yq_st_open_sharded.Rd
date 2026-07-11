% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_st_open_sharded}
\alias{yq_st_open_sharded}
\title{Open a sharded safetensors directory for partial reads}
\usage{
yq_st_open_sharded(dir)
}
\arguments{
\item{dir}{Directory containing the index and shards.}
}
\value{
An opaque sharded handle.
}
\description{
Handles the \code{model.safetensors.index.json} +
\code{model-000NN-of-000MM.safetensors} layout. Returns a handle
whose \code{$header} spans all shards; \code{\link{yq_st_read}}
dispatches each key to its shard. Close with \code{\link{yq_st_close}}.
}
