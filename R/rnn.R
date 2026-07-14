#' Recurrent layers for anvl
#'
#' A torch-compatible LSTM built from anvl primitives. The recurrence is
#' unrolled over time at trace time (fine for the bounded sequences these
#' ports use, e.g. speaker-embedding partials); a rolled \code{nv_while}
#' version is a future optimization for long sequences. Weights use the
#' \code{\link{linear}} layout: \code{w_ih} is \code{[input, 4 * hidden]}
#' and \code{w_hh} is \code{[hidden, 4 * hidden]} (the transpose of torch's
#' \code{[4 * hidden, .]} \code{weight_ih_l*} / \code{weight_hh_l*}, which
#' \code{\link{st_read}} produces for free). Gate order is torch's:
#' input, forget, cell, output.
#'
#' @name rnn
NULL

#' One LSTM timestep
#'
#' @param x_t AnvlArray \code{[B, input]}.
#' @param h AnvlArray \code{[B, hidden]}. Previous hidden state.
#' @param c AnvlArray \code{[B, hidden]}. Previous cell state.
#' @param w_ih AnvlArray \code{[input, 4 * hidden]}.
#' @param w_hh AnvlArray \code{[hidden, 4 * hidden]}.
#' @param b_ih AnvlArray \code{[4 * hidden]} or NULL.
#' @param b_hh AnvlArray \code{[4 * hidden]} or NULL.
#'
#' @return List with \code{h} and \code{c}, each \code{[B, hidden]}.
#'
#' @export
lstm_cell <- function(x_t, h, c, w_ih, w_hh, b_ih = NULL, b_hh = NULL) {
    gates <- linear(x_t, w_ih, b_ih) + linear(h, w_hh, b_hh)
    hd <- anvl::shape(h)[2L]
    i <- anvl::nv_logistic(slice_lastdim(gates, 1L, hd))
    f <- anvl::nv_logistic(slice_lastdim(gates, hd + 1L, 2L * hd))
    g <- anvl::nv_tanh(slice_lastdim(gates, 2L * hd + 1L, 3L * hd))
    o <- anvl::nv_logistic(slice_lastdim(gates, 3L * hd + 1L, 4L * hd))
    c_new <- f * c + i * g
    list(h = o * anvl::nv_tanh(c_new), c = c_new)
}

#' Multi-layer LSTM forward pass
#'
#' Stacked unidirectional LSTM matching \code{torch::nn_lstm} with
#' \code{bidirectional = FALSE}. Each layer feeds its full output sequence
#' to the next.
#'
#' @param x AnvlArray. \code{[seq, batch, input]}, or
#'   \code{[batch, seq, input]} when \code{batch_first = TRUE}.
#' @param layers List of per-layer parameter lists, each with elements
#'   \code{w_ih}, \code{w_hh}, \code{b_ih}, \code{b_hh} (biases may be
#'   NULL). Layer 1's \code{input} is \code{x}'s feature size; deeper
#'   layers' is the previous \code{hidden}.
#' @param batch_first Logical. Layout of \code{x} and the returned
#'   \code{output}.
#'
#' @return List with \code{output} (all top-layer hidden states, same
#'   layout as \code{x}), \code{h_n} and \code{c_n}
#'   (\code{[num_layers, batch, hidden]}, each layer's final state).
#'
#' @export
lstm <- function(x, layers, batch_first = FALSE) {
    if (batch_first) {
        x <- anvl::nv_transpose(x, c(2L, 1L, 3L))
    }
    s <- anvl::shape(x)
    n_seq <- s[1L]
    batch <- s[2L]
    dt <- anvl::dtype(x)
    h_n <- vector("list", length(layers))
    c_n <- vector("list", length(layers))
    for (l in seq_along(layers)) {
        ly <- layers[[l]]
        hd <- anvl::shape(ly$w_hh)[1L]
        h <- anvl::nv_fill(0, shape = c(batch, hd), dtype = dt)
        cc <- anvl::nv_fill(0, shape = c(batch, hd), dtype = dt)
        indim <- anvl::shape(x)[3L]
        outs <- vector("list", n_seq)
        for (t in seq_len(n_seq)) {
            x_t <- anvl::nv_reshape(
                anvl::nv_static_slice(x,
                    start_indices = c(t, 1L, 1L),
                    limit_indices = c(t, batch, indim),
                    strides = c(1L, 1L, 1L)),
                c(batch, indim))
            step <- lstm_cell(x_t, h, cc, ly$w_ih, ly$w_hh, ly$b_ih, ly$b_hh)
            h <- step$h
            cc <- step$c
            outs[[t]] <- anvl::nv_unsqueeze(h, 1L)
        }
        x <- do.call(anvl::nv_concatenate, c(outs, list(dimension = 1L)))
        h_n[[l]] <- anvl::nv_unsqueeze(h, 1L)
        c_n[[l]] <- anvl::nv_unsqueeze(cc, 1L)
    }
    output <- if (batch_first) anvl::nv_transpose(x, c(2L, 1L, 3L)) else x
    list(
        output = output,
        h_n = do.call(anvl::nv_concatenate, c(h_n, list(dimension = 1L))),
        c_n = do.call(anvl::nv_concatenate, c(c_n, list(dimension = 1L)))
    )
}
