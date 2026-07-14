% tinyrox says don't edit this manually, but it can't stop you!
\name{conv}
\alias{conv}
\title{Convolution family for anvl}
\description{
Torch-layout wrappers over anvl's \code{nv_conv*} primitives, adding the
optional bias that \code{nv_conv*} omits, plus a transposed 1-D
convolution built from interior-padding + a plain conv (anvl has no
native \code{conv_transpose}). All use NCW / NCHW / NCDHW layout and the
torch weight conventions.
}
