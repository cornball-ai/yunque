% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_st_open}
\alias{yq_st_open}
\title{Open a safetensors file for partial reads}
\usage{
yq_st_open(path)
}
\arguments{
\item{path}{Path to a .safetensors file.}
}
\value{
An opaque handle (a list with the open connection, parsed
  header, and data offset).
}
\description{
Returns a handle for \code{\link{yq_st_read}}. Pair with
\code{\link{yq_st_close}}.
}
