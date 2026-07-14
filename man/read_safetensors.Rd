% tinyrox says don't edit this manually, but it can't stop you!
\name{read_safetensors}
\alias{read_safetensors}
\title{Read safetensors tensors as R arrays (base R, partial reads)}
\usage{
read_safetensors(path, keys = NULL, transpose_2d = FALSE)
}
\arguments{
\item{path}{Path to a .safetensors file.}

\item{keys}{Character vector of tensor names, or NULL for all.}

\item{transpose_2d}{Logical. Return 2-D tensors transposed (free; a
\code{[out, in]} checkpoint matrix becomes the \code{[in, out]}
layout \code{\link{linear}} wants).}
}
\value{
Named list of R arrays/vectors (doubles, exact f32 values).
}
\description{
Convenience wrapper over \code{\link{st_read}}: F16/BF16/F32
payloads, no torch dependency, reads only the requested keys.
}
