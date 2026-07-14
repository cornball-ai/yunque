% tinyrox says don't edit this manually, but it can't stop you!
\name{st_read}
\alias{st_read}
\title{Read one tensor from a safetensors handle as an R array}
\usage{
st_read(st, key, transpose = FALSE)
}
\arguments{
\item{st}{A handle from \code{\link{st_open}} /
\code{\link{st_open_sharded}}.}

\item{key}{Tensor name.}

\item{transpose}{Logical. Return 2-D tensors transposed, which is free
— a row-major \code{[out, in]} checkpoint matrix read column-major
already IS the \code{[in, out]} logical transpose (what a linear
layer wants).}
}
\value{
An R array / vector of doubles (exact f32 values).
}
\description{
F16/BF16 upcast to f32 (BF16 is the top half of an f32; F16 is
true IEEE half, converted per-element). Reading
a single key seeks straight to its bytes, so pulling a few tensors out
of a multi-GB checkpoint is cheap.
}
