% tinyrox says don't edit this manually, but it can't stop you!
\name{timestep_embedding}
\alias{timestep_embedding}
\title{Sinusoidal timestep embedding}
\usage{
timestep_embedding(
  timesteps,
  dim,
  max_period = 10000,
  flip_sin_to_cos = FALSE,
  downscale_freq_shift = 1,
  scale = 1,
  dtype = "f32"
)
}
\arguments{
\item{timesteps}{AnvlArray \code{[B]}, or a numeric vector.}

\item{dim}{Integer. Embedding width.}

\item{max_period}{Numeric. Longest period (default 10000).}

\item{flip_sin_to_cos}{Logical. Put cos before sin.}

\item{downscale_freq_shift}{Numeric. Subtracted from \code{half} in the
frequency denominator.}

\item{scale}{Numeric. Multiplies the angles before sin and cos.}

\item{dtype}{Character. Dtype used when \code{timesteps} is a plain
numeric vector; an AnvlArray keeps its own.}
}
\value{
AnvlArray \code{[B, dim]}.
}
\description{
The diffusers \code{get_timestep_embedding()}: with
\code{half = dim \%/\% 2}, frequencies
\code{exp(-log(max_period) * i / (half - downscale_freq_shift))} for
\code{i} in \code{0:(half - 1)}, then \code{[sin(t f), cos(t f)]} per
timestep, or cos first with \code{flip_sin_to_cos}. FLUX and SDXL use
\code{flip_sin_to_cos = TRUE, downscale_freq_shift = 0}; the DDPM
default is \code{FALSE, 1}. An odd \code{dim} is zero-padded by one
column. The frequencies are a constant at trace time, so inside
\code{anvl::jit()} only the outer product and the trig run on device.
}
