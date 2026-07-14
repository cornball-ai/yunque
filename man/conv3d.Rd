% tinyrox says don't edit this manually, but it can't stop you!
\name{conv3d}
\alias{conv3d}
\title{3-D convolution with optional bias}
\usage{
conv3d(input, weight, bias = NULL, stride = 1L, padding = 0L, dilation = 1L,
       groups = 1L)
}
\arguments{
\item{input}{AnvlArray \code{[N, C_in, D, H, W]}.}

\item{weight}{AnvlArray \code{[C_out, C_in / groups, kD, kH, kW]}.}

\item{bias}{AnvlArray \code{[C_out]} or NULL.}

\item{stride}{Integer (length 1 or 3). Convolution stride.}

\item{padding}{Integer (length 1 or 3). Symmetric zero-padding.}

\item{dilation}{Integer (length 1 or 3). Kernel dilation.}

\item{groups}{Integer.}
}
\description{
NCDHW layout: \code{input} \code{[N, C_in, D, H, W]}, \code{weight}
\code{[C_out, C_in / groups, kD, kH, kW]}, optional \code{bias}
\code{[C_out]}.
}
