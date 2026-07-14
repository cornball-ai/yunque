% tinyrox says don't edit this manually, but it can't stop you!
\name{upsample_nearest2d}
\alias{upsample_nearest2d}
\title{Nearest-neighbour 2x upsampling over [B, C, H, W]}
\usage{
upsample_nearest2d(x)
}
\arguments{
\item{x}{AnvlArray \code{[B, C, H, W]}.}
}
\description{
Each pixel becomes a 2x2 block (matches
\code{nnf_interpolate(scale_factor = 2, mode = "nearest")}).
}
