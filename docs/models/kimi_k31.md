# Kimi K3.1 (packed MXFP4)

| GPU | Tensor parallel | Coverage |
| --- | --- | --- |
| GB300 (SM103); 148-SM B300 (SM103) and B200/GB200 (SM100) with `sm148` | 16 (one NVLink domain); 8 | DFlash target verify steps of 8 rows a request (7 drafted tokens and the anchor) at 1 to 8 requests, without DCP or with DCP equal to TP, also as the decode role of a PD-disaggregated deployment; the DFlash2 draft decoder layers at one request. Served and measured end to end at TP16, one request; on GB200 (TP16 over four trays, `sm148`) served and measured end to end at one to four requests |

The Kimi K3.1 megakernel (`sgl_nvidia_mega/kimi_k31`) runs the DFlash target verify
step and the DFlash2 draft's decoder layers. It needs an SGLang with the Kimi K3.1
host bindings (sgl-project/sglang#48414); on an older SGLang its spec refuses every
configuration with that reason. It builds on its own
copy of the Kimi K3 base (`kimi_k31/k3` and its kernels in `kimi_k31/sm103/k3`), so
`sgl_nvidia_mega/kimi_k3` and the Kimi K3 path are unchanged.
