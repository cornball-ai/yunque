% tinyrox says don't edit this manually, but it can't stop you!
\name{load_weights}
\alias{load_weights}
\title{Load a weight tree from a safetensors checkpoint}
\usage{
load_weights(
  st,
  spec,
  dtype = "f32",
  device = "cpu",
  unused = c("warn", "error", "ignore")
)
}
\arguments{
\item{st}{A handle from \code{\link{st_open}} or
\code{\link{st_open_sharded}}, or a path to a .safetensors file,
opened and closed here.}

\item{spec}{Nested list of weight specs.}

\item{dtype}{Character. Device dtype for every leaf.}

\item{device}{Character.}

\item{unused}{Character. What to do about checkpoint keys the spec does
not read: \code{"warn"} (default), \code{"error"} or \code{"ignore"}.}
}
\value{
The spec tree with \code{AnvlArray} leaves.
}
\description{
A port's weights are a nested named list mirroring the model: blocks,
then layers, then the tensors each layer needs. Instead of a loader per
model that opens the file, reads each key, transposes matrices and
uploads them, describe the tree once with \code{\link{w_linear}},
\code{\link{w_tensor}}, \code{\link{w_cat}} and \code{\link{w_layers}}
and let this walk it. Each leaf is read with \code{\link{st_read}} and
returned as an \code{AnvlArray} on \code{device}. Non-spec values in
the tree (\code{NULL}, numbers, strings) pass through, so static
configuration can sit next to the weights it describes.
}
\details{
Every key named in the spec must exist, an error otherwise. Keys in
the checkpoint the spec never reads are reported according to
\code{unused}, so a port can prove its census: every tensor the file
ships has a slot. They are also attached as
\code{attr(result, "unused_keys")}.

}
