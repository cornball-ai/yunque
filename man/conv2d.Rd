% tinyrox says don't edit this manually, but it can't stop you!
\name{conv2d}
\alias{conv2d}
\title{2-D convolution with optional bias}
\usage{
conv2d(input, weight, bias = NULL, stride = 1L, padding = 0L, dilation = 1L,
       groups = 1L)
}
\arguments{
\item{input}{AnvlArray \code{[N, C_in, H, W]}.}

\item{weight}{AnvlArray \code{[C_out, C_in / groups, kH, kW]}.}

\item{bias}{AnvlArray \code{[C_out]} or NULL.}

\item{stride}{Integer (length 1 or 2). Convolution stride.}

\item{padding}{Integer (length 1 or 2). Symmetric zero-padding.}

\item{dilation}{Integer (length 1 or 2). Kernel dilation.}

\item{groups}{Integer.}
}
\description{
NCHW layout: \code{input} \code{[N, C_in, H, W]}, \code{weight}
\code{[C_out, C_in / groups, kH, kW]}, optional \code{bias}
\code{[C_out]}.
}
