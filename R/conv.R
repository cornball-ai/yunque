#' Convolution family for anvl
#'
#' Torch-layout wrappers over anvl's \code{nv_conv*} primitives, adding the
#' optional bias that \code{nv_conv*} omits, plus a transposed 1-D
#' convolution built from interior-padding + a plain conv (anvl has no
#' native \code{conv_transpose}). All use NCW / NCHW / NCDHW layout and the
#' torch weight conventions.
#'
#' @name conv
NULL

# Add an optional [C_out] bias to a conv output [N, C_out, *spatial].
.conv_add_bias <- function(out, bias) {
    if (is.null(bias)) {
        return(out)
    }
    s <- anvl::shape(out)
    cshape <- rep(1L, length(s))
    cshape[2L] <- s[2L]
    out + anvl::nv_broadcast_to(anvl::nv_reshape(bias, cshape), s)
}

#' 1-D convolution with optional bias
#'
#' NCW layout: \code{input} \code{[N, C_in, W]}, \code{weight}
#' \code{[C_out, C_in / groups, kW]}, optional \code{bias} \code{[C_out]}.
#' Thin wrapper over \code{\link[anvl]{nv_conv1d}}.
#'
#' @param input AnvlArray \code{[N, C_in, W]}.
#' @param weight AnvlArray \code{[C_out, C_in / groups, kW]}.
#' @param bias AnvlArray \code{[C_out]} or NULL.
#' @param stride Integer. Convolution stride.
#' @param padding Integer. Symmetric zero-padding on each side.
#' @param dilation Integer. Spacing between kernel taps.
#' @param groups Integer. Grouped/depthwise convolution.
#'
#' @export
conv1d <- function(input, weight, bias = NULL, stride = 1L, padding = 0L,
                   dilation = 1L, groups = 1L) {
    out <- anvl::nv_conv1d(input, weight, stride = stride, padding = padding,
                           dilation = dilation, groups = groups)
    .conv_add_bias(out, bias)
}

#' 2-D convolution with optional bias
#'
#' NCHW layout: \code{input} \code{[N, C_in, H, W]}, \code{weight}
#' \code{[C_out, C_in / groups, kH, kW]}, optional \code{bias}
#' \code{[C_out]}.
#'
#' @param input AnvlArray \code{[N, C_in, H, W]}.
#' @param weight AnvlArray \code{[C_out, C_in / groups, kH, kW]}.
#' @param bias AnvlArray \code{[C_out]} or NULL.
#' @param stride Integer (length 1 or 2). Convolution stride.
#' @param padding Integer (length 1 or 2). Symmetric zero-padding.
#' @param dilation Integer (length 1 or 2). Kernel dilation.
#' @param groups Integer.
#'
#' @export
conv2d <- function(input, weight, bias = NULL, stride = 1L, padding = 0L,
                   dilation = 1L, groups = 1L) {
    out <- anvl::nv_conv2d(input, weight, stride = stride, padding = padding,
                           dilation = dilation, groups = groups)
    .conv_add_bias(out, bias)
}

#' 3-D convolution with optional bias
#'
#' NCDHW layout: \code{input} \code{[N, C_in, D, H, W]}, \code{weight}
#' \code{[C_out, C_in / groups, kD, kH, kW]}, optional \code{bias}
#' \code{[C_out]}.
#'
#' @param input AnvlArray \code{[N, C_in, D, H, W]}.
#' @param weight AnvlArray \code{[C_out, C_in / groups, kD, kH, kW]}.
#' @param bias AnvlArray \code{[C_out]} or NULL.
#' @param stride Integer (length 1 or 3). Convolution stride.
#' @param padding Integer (length 1 or 3). Symmetric zero-padding.
#' @param dilation Integer (length 1 or 3). Kernel dilation.
#' @param groups Integer.
#'
#' @export
conv3d <- function(input, weight, bias = NULL, stride = 1L, padding = 0L,
                   dilation = 1L, groups = 1L) {
    out <- anvl::nv_conv3d(input, weight, stride = stride, padding = padding,
                           dilation = dilation, groups = groups)
    .conv_add_bias(out, bias)
}

#' 1-D transposed convolution with optional bias
#'
#' Matches \code{torch::nnf_conv_transpose1d}. Note the torch
#' ConvTranspose weight layout is \emph{in-channels first}:
#' \code{[C_in, C_out, kW]}. anvl has no native transposed conv, so this
#' builds it from the standard equivalence -- dilate the input by
#' \code{stride} (interior zeros), pad, then a plain conv with the kernel
#' spatially flipped and its in/out channels swapped. \code{groups = 1}
#' only (covers the HiFiGAN-style upsamplers that need it).
#'
#' @param input AnvlArray \code{[N, C_in, L]}.
#' @param weight AnvlArray \code{[C_in, C_out, kW]}.
#' @param bias AnvlArray \code{[C_out]} or NULL.
#' @param stride Integer. Upsampling factor.
#' @param padding Integer. Trimmed symmetrically from the output.
#' @param output_padding Integer. Extra size added to the high side.
#' @param dilation Integer. Kernel dilation.
#' @param groups Integer. Must be 1.
#'
#' @export
conv_transpose1d <- function(input, weight, bias = NULL, stride = 1L,
                             padding = 0L, output_padding = 0L,
                             dilation = 1L, groups = 1L) {
    if (groups != 1L) {
        stop("conv_transpose1d supports groups = 1 only")
    }
    stride <- as.integer(stride)
    padding <- as.integer(padding)
    output_padding <- as.integer(output_padding)
    dilation <- as.integer(dilation)
    k <- anvl::shape(weight)[3L]
    lo <- dilation * (k - 1L) - padding
    xp <- anvl::nv_pad(
        input, 0,
        edge_padding_low = c(0L, 0L, lo),
        edge_padding_high = c(0L, 0L, lo + output_padding),
        interior_padding = c(0L, 0L, stride - 1L)
    )
    w <- anvl::nv_reverse(anvl::nv_transpose(weight, c(2L, 1L, 3L)), dims = 3L)
    out <- anvl::nv_conv1d(xp, w, stride = 1L, padding = 0L,
                           dilation = dilation, groups = 1L)
    .conv_add_bias(out, bias)
}
