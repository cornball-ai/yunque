% tinyrox says don't edit this manually, but it can't stop you!
\name{conv_transpose1d}
\alias{conv_transpose1d}
\title{1-D transposed convolution with optional bias}
\usage{
conv_transpose1d(input, weight, bias = NULL, stride = 1L, padding = 0L,
                 output_padding = 0L, dilation = 1L, groups = 1L)
}
\arguments{
\item{input}{AnvlArray \code{[N, C_in, L]}.}

\item{weight}{AnvlArray \code{[C_in, C_out, kW]}.}

\item{bias}{AnvlArray \code{[C_out]} or NULL.}

\item{stride}{Integer. Upsampling factor.}

\item{padding}{Integer. Trimmed symmetrically from the output.}

\item{output_padding}{Integer. Extra size added to the high side.}

\item{dilation}{Integer. Kernel dilation.}

\item{groups}{Integer. Must be 1.}
}
\description{
Matches \code{torch::nnf_conv_transpose1d}. Note the torch
ConvTranspose weight layout is \emph{in-channels first}:
\code{[C_in, C_out, kW]}. anvl has no native transposed conv, so this
builds it from the standard equivalence -- dilate the input by
\code{stride} (interior zeros), pad, then a plain conv with the kernel
spatially flipped and its in/out channels swapped. \code{groups = 1}
only (covers the HiFiGAN-style upsamplers that need it).
}
