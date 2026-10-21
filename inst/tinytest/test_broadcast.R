# Broadcasting arithmetic vs base-R sweep() references (needs anvl + PJRT
# CPU).

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

nva <- function(a) anvl::nv_array(c(a), dtype = "f32", shape = dim(a))

set.seed(11)
x_r <- array(rnorm(2 * 3 * 4), c(2, 3, 4))
x <- nva(x_r)
v_r <- rnorm(4)
v <- anvl::nv_array(v_r, dtype = "f32")

# [B, S, D] + [D]: right-aligned rank expansion
got <- as.array(badd(x, v))
expect_equal(dim(got), c(2L, 3L, 4L))
expect_true(max(abs(got - sweep(x_r, 3, v_r, "+"))) < 1e-6)

# [B, S, D] * [B, 1, D]: size-1 axis expansion
g_r <- array(rnorm(2 * 4), c(2, 1, 4))
ref <- x_r * array(g_r[, rep(1, 3), , drop = FALSE], c(2, 3, 4))
expect_true(max(abs(as.array(bmul(x, nva(g_r))) - ref)) < 1e-6)

# [D] on the left broadcasts the same way
expect_true(max(abs(as.array(bsub(v, x)) - sweep(-x_r, 3, v_r, "+"))) < 1e-6)

# plain R numbers pass straight through
expect_true(max(abs(as.array(bsub(x, 1)) - (x_r - 1))) < 1e-6)
expect_true(max(abs(as.array(bdiv(x, 2)) - (x_r / 2))) < 1e-6)
expect_true(max(abs(as.array(bdiv(2, x)) - (2 / x_r))) < 1e-5)

# shapes numpy would refuse are refused
expect_error(badd(x, anvl::nv_array(rnorm(5), dtype = "f32")))

# works under jit
f <- anvl::jit(function(a, b) bmul(a, b))
expect_true(max(abs(as.array(f(x, v)) - sweep(x_r, 3, v_r, "*"))) < 1e-6)
