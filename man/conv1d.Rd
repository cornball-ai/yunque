% tinyrox says don't edit this manually, but it can't stop you!
\name{conv1d}
\alias{conv1d}
\title{1-D convolution with optional bias}
\usage{
conv1d(input, weight, bias = NULL, stride = 1L, padding = 0L, dilation = 1L,
       groups = 1L)
}
\arguments{
\item{input}{AnvlArray \code{[N, C_in, W]}.}

\item{weight}{AnvlArray \code{[C_out, C_in / groups, kW]}.}

\item{bias}{AnvlArray \code{[C_out]} or NULL.}

\item{stride}{Integer. Convolution stride.}

\item{padding}{Integer. Symmetric zero-padding on each side.}

\item{dilation}{Integer. Spacing between kernel taps.}

\item{groups}{Integer. Grouped/depthwise convolution.}
}
\description{
NCW layout: \code{input} \code{[N, C_in, W]}, \code{weight}
\code{[C_out, C_in / groups, kW]}, optional \code{bias} \code{[C_out]}.
Thin wrapper over \code{\link[anvl]{nv_conv1d}}.
}
