# STFT/ISTFT vs base-R DFT + round-trip (needs anvl + PJRT CPU).

if (!requireNamespace("anvl", quietly = TRUE)) {
    exit_file("anvl not installed")
}

reltol <- function(got, ref) max(abs(got - ref)) / max(abs(ref))

# base-R reference matching torch.stft (one-sided, periodic window,
# center reflect padding). sig [B, samples] -> list(re, im) [B, nf, frames].
ref_stft <- function(sig, n_fft, hop, win, center) {
    if (center) {
        p <- n_fft %/% 2
        sig <- t(apply(sig, 1, function(r) {
            c(rev(r[2:(p + 1)]), r, rev(r[(length(r) - p):(length(r) - 1)]))
        }))
    }
    B <- nrow(sig)
    L <- ncol(sig)
    nf <- n_fft %/% 2 + 1
    frames <- 1 + (L - n_fft) %/% hop
    re <- array(0, c(B, nf, frames))
    im <- array(0, c(B, nf, frames))
    n <- 0:(n_fft - 1)
    for (b in 1:B) {
        for (m in 0:(frames - 1)) {
            fr <- sig[b, (m * hop + 1):(m * hop + n_fft)] * win
            for (k in 0:(nf - 1)) {
                re[b, k + 1, m + 1] <- sum(fr * cos(2 * pi * k * n / n_fft))
                im[b, k + 1, m + 1] <- -sum(fr * sin(2 * pi * k * n / n_fft))
            }
        }
    }
    list(re = re, im = im)
}

set.seed(5)
n_fft <- 16L
hop <- 4L
B <- 2L
samp <- 48L
sig <- matrix(rnorm(B * samp), B, samp)
win <- hann_window(n_fft)

# forward STFT (Hann, center) vs reference
sp <- stft(anvl::nv_array(sig, dtype = "f32"), n_fft, hop, window = win, center = TRUE)
ref <- ref_stft(sig, n_fft, hop, win, TRUE)
expect_equal(dim(as.array(sp$real)), c(B, n_fft %/% 2L + 1L, dim(ref$re)[3]))
expect_true(reltol(as.array(sp$real), ref$re) < 1e-4)
expect_true(reltol(as.array(sp$imag), ref$im) < 1e-4)

# rectangular window (NULL) + center = FALSE
sp2 <- stft(anvl::nv_array(sig, dtype = "f32"), n_fft, hop, window = NULL, center = FALSE)
ref2 <- ref_stft(sig, n_fft, hop, rep(1, n_fft), FALSE)
expect_true(reltol(as.array(sp2$real), ref2$re) < 1e-4)
expect_true(reltol(as.array(sp2$imag), ref2$im) < 1e-4)

# ISTFT round-trip recovers the signal (COLA Hann window)
rec <- as.array(istft(sp$real, sp$imag, n_fft, hop, window = win,
                      center = TRUE, length = samp))
expect_equal(dim(rec), c(B, samp))
expect_true(reltol(rec, sig) < 1e-4)

# hann_window is the periodic form (endpoints not both zero-symmetric)
hw <- hann_window(4L)
expect_equal(hw, c(0, 0.5, 1, 0.5))
