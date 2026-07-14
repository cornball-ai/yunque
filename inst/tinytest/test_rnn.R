# LSTM vs a base-R reference (needs anvl + PJRT CPU).

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

nva <- function(a) anvl::nv_array(c(a), dtype = "f32", shape = dim(a))
sig <- function(z) 1 / (1 + exp(-z))

# base-R stacked LSTM. x [S, B, In]; layers use torch layout:
# Wih [4H, In], Whh [4H, H], bih/bhh [4H]. Returns list(output, h_n).
ref_lstm <- function(x, layers) {
    S <- dim(x)[1]
    B <- dim(x)[2]
    h_n <- list()
    for (li in seq_along(layers)) {
        ly <- layers[[li]]
        H <- nrow(ly$Whh) / 4L
        Inn <- dim(x)[3]
        h <- matrix(0, B, H)
        cc <- matrix(0, B, H)
        out <- array(0, c(S, B, H))
        for (t in seq_len(S)) {
            xt <- matrix(x[t, , ], B, Inn)
            g <- xt %*% t(ly$Wih) + matrix(ly$bih, B, 4 * H, byrow = TRUE) +
                h %*% t(ly$Whh) + matrix(ly$bhh, B, 4 * H, byrow = TRUE)
            i <- sig(g[, 1:H, drop = FALSE])
            f <- sig(g[, (H + 1):(2 * H), drop = FALSE])
            gg <- tanh(g[, (2 * H + 1):(3 * H), drop = FALSE])
            o <- sig(g[, (3 * H + 1):(4 * H), drop = FALSE])
            cc <- f * cc + i * gg
            h <- o * tanh(cc)
            out[t, , ] <- h
        }
        x <- out
        h_n[[li]] <- h
    }
    list(output = x, h_n = h_n)
}

reltol <- function(got, ref) max(abs(got - ref)) / max(abs(ref))

set.seed(11)
S <- 6L
B <- 2L
In <- 4L
H <- 5L
sizes <- list(c(In, H), c(H, H)) # a 2-layer stack: In->H, H->H

x <- array(rnorm(S * B * In), c(S, B, In))
mk_layer <- function(inn, hh) {
    list(Wih = matrix(rnorm(4 * hh * inn), 4 * hh, inn),
         Whh = matrix(rnorm(4 * hh * hh), 4 * hh, hh),
         bih = rnorm(4 * hh), bhh = rnorm(4 * hh))
}
tlayers <- lapply(sizes, function(sz) mk_layer(sz[1], sz[2]))

ref <- ref_lstm(x, tlayers)

# yunque layout: w_ih = t(Wih) [In, 4H], w_hh = t(Whh) [H, 4H]
ylayers <- lapply(tlayers, function(ly) list(
    w_ih = nva(t(ly$Wih)), w_hh = nva(t(ly$Whh)),
    b_ih = anvl::nv_array(ly$bih, dtype = "f32"),
    b_hh = anvl::nv_array(ly$bhh, dtype = "f32")))

res <- lstm(nva(x), ylayers)

# output sequence [S, B, H] matches
expect_equal(dim(as.array(res$output)), c(S, B, H))
expect_true(reltol(as.array(res$output), ref$output) < 1e-4)

# final hidden state per layer [num_layers, B, H]
hn <- as.array(res$h_n)
expect_equal(dim(hn), c(2L, B, H))
expect_true(reltol(hn[1, , ], ref$h_n[[1]]) < 1e-4)
expect_true(reltol(hn[2, , ], ref$h_n[[2]]) < 1e-4)

# batch_first: [B, S, In] input gives [B, S, H] output, same values
xbf <- aperm(x, c(2, 1, 3))
resbf <- lstm(nva(xbf), ylayers, batch_first = TRUE)
expect_equal(dim(as.array(resbf$output)), c(B, S, H))
expect_true(reltol(aperm(as.array(resbf$output), c(2, 1, 3)), ref$output) < 1e-4)

# single-step lstm_cell matches one manual step of layer 1
h0 <- matrix(0, B, H)
c0 <- matrix(0, B, H)
ly1 <- tlayers[[1]]
g <- matrix(x[1, , ], B, In) %*% t(ly1$Wih) + matrix(ly1$bih, B, 4 * H, byrow = TRUE) +
    h0 %*% t(ly1$Whh) + matrix(ly1$bhh, B, 4 * H, byrow = TRUE)
i <- sig(g[, 1:H]); f <- sig(g[, (H + 1):(2 * H)])
gg <- tanh(g[, (2 * H + 1):(3 * H)]); o <- sig(g[, (3 * H + 1):(4 * H)])
c_ref <- f * c0 + i * gg
h_ref <- o * tanh(c_ref)
step <- lstm_cell(nva(matrix(x[1, , ], B, In)),
                  anvl::nv_array(h0, dtype = "f32"), anvl::nv_array(c0, dtype = "f32"),
                  ylayers[[1]]$w_ih, ylayers[[1]]$w_hh,
                  ylayers[[1]]$b_ih, ylayers[[1]]$b_hh)
expect_true(reltol(as.array(step$h), h_ref) < 1e-4)
expect_true(reltol(as.array(step$c), c_ref) < 1e-4)
