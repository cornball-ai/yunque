% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_st_read}
\alias{yq_st_read}
\title{Read one tensor from a safetensors handle as an R array}
\usage{
yq_st_read(st, key, transpose = FALSE)
}
\arguments{
\item{st}{A handle from \code{\link{yq_st_open}} /
\code{\link{yq_st_open_sharded}}.}

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
BF16 upcasts to f32 exactly (bf16 is the top half of an f32). Reading
a single key seeks straight to its bytes, so pulling a few tensors out
of a multi-GB checkpoint is cheap.
}
