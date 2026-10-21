# Declarative weight loading against a hand-written safetensors file
# (needs anvl + PJRT CPU).

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

# Write F32 tensors the way torch does: row-major payloads, shapes as
# stored. A 2-D R matrix M becomes the [nrow, ncol] tensor with bytes
# c(t(M)); st_read() then returns M (or t(M) with transpose = TRUE).
write_st <- function(path, tensors) {
    header <- list()
    blobs <- list()
    offset <- 0L
    for (nm in names(tensors)) {
        a <- tensors[[nm]]
        d <- dim(a)
        if (is.null(d)) {
            d <- length(a)
            vals <- a
        } else if (length(d) == 2L) {
            vals <- c(t(a))
        } else {
            vals <- c(aperm(a, rev(seq_along(d))))
        }
        nbytes <- length(vals) * 4L
        header[[nm]] <- list(dtype = "F32", shape = as.list(as.integer(d)),
                             data_offsets = list(offset, offset + nbytes))
        blobs[[nm]] <- vals
        offset <- offset + nbytes
    }
    hj <- charToRaw(as.character(jsonlite::toJSON(header, auto_unbox = TRUE)))
    con <- file(path, "wb")
    on.exit(close(con))
    writeBin(length(hj), con, size = 8L, endian = "little")
    writeBin(hj, con)
    for (v in blobs) {
        writeBin(as.numeric(v), con, size = 4L, endian = "little")
    }
}

set.seed(5)
Wq <- matrix(rnorm(6 * 4), 6, 4) # torch [out = 6, in = 4]
Wk <- matrix(rnorm(6 * 4), 6, 4)
Wv <- matrix(rnorm(6 * 4), 6, 4)
bq <- rnorm(6)
bk <- rnorm(6)
bv <- rnorm(6)
norm_w <- rnorm(4)
tensors <- list(
    "blocks.0.attn.q.weight" = Wq,
    "blocks.0.attn.k.weight" = Wk,
    "blocks.0.attn.v.weight" = Wv,
    "blocks.0.attn.q.bias" = bq,
    "blocks.0.attn.k.bias" = bk,
    "blocks.0.attn.v.bias" = bv,
    "blocks.0.norm.weight" = norm_w,
    "blocks.1.norm.weight" = norm_w * 2,
    "unused.thing" = rnorm(3)
)
path <- tempfile(fileext = ".safetensors")
write_st(path, tensors)

spec <- list(
    blocks = w_layers(2L, function(i) {
        p <- sprintf("blocks.%d.", i - 1L)
        attn <- NULL
        if (i == 1L) {
            attn <- list(
                qkv = w_cat(paste0(p, "attn.", c("q", "k", "v"), ".weight"),
                            dim = 2L),
                qkv_bias = w_cat(paste0(p, "attn.", c("q", "k", "v"), ".bias"),
                                 dim = 1L, transpose = FALSE),
                q = w_linear(paste0(p, "attn.q.weight"))
            )
        }
        list(norm = w_tensor(paste0(p, "norm.weight")), attn = attn,
             heads = 2L)
    })
)

# unused keys warn by default and are reported on the result
expect_warning(w <- load_weights(path, spec), "not read")
expect_equal(attr(w, "unused_keys"), "unused.thing")

# tree shape mirrors the spec; non-spec values pass through
expect_equal(length(w$blocks), 2L)
expect_equal(w$blocks[[1]]$heads, 2L)
expect_null(w$blocks[[2]]$attn)
expect_true(inherits(w$blocks[[1]]$norm, "AnvlArray"))

# w_linear returns [in, out]
expect_true(max(abs(as.array(w$blocks[[1]]$attn$q) - t(Wq))) < 1e-6)

# w_cat fuses projections column-wise and biases end to end
expect_true(max(abs(as.array(w$blocks[[1]]$attn$qkv) -
                    cbind(t(Wq), t(Wk), t(Wv)))) < 1e-6)
expect_true(max(abs(as.array(w$blocks[[1]]$attn$qkv_bias) - c(bq, bk, bv))) <
            1e-6)
expect_true(max(abs(as.array(w$blocks[[2]]$norm) - norm_w * 2)) < 1e-6)

# the fused weight drives linear() like the three projections side by side
x_r <- matrix(rnorm(3 * 4), 3, 4)
got <- as.array(linear(anvl::nv_array(x_r, dtype = "f32"),
                       w$blocks[[1]]$attn$qkv))
expect_true(max(abs(got - x_r %*% cbind(t(Wq), t(Wk), t(Wv)))) < 1e-5)

# unused = "error" / "ignore"
expect_error(load_weights(path, spec, unused = "error"), "not read")
expect_silent(w3 <- load_weights(path, spec, unused = "ignore"))
expect_equal(attr(w3, "unused_keys"), "unused.thing")

# a key missing from the file is an error
expect_error(load_weights(path, list(w_tensor("nope")), unused = "ignore"),
             "key not in file")

# an open handle works as well as a path
st <- st_open(path)
w2 <- load_weights(st, spec, unused = "ignore")
st_close(st)
expect_equal(as.array(w2$blocks[[1]]$norm), as.array(w$blocks[[1]]$norm))
