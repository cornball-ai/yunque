% tinyrox says don't edit this manually, but it can't stop you!
\name{ops}
\alias{ops}
\title{Core neural-network ops for anvl}
\description{
anvl binary operators broadcast scalars only; every reduction that
feeds a binary op needs an explicit \code{nv_broadcast_to}. These
helpers package the broadcast dance behind torch-like semantics.
All functions work on \code{AnvlArray}s and inside \code{anvl::jit()}.
}
