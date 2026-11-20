# Attention-residual auxiliary mixture

The original `PreparedAttnRes.bind_aux()` publishes the updated raw prefix.
DSpark instead consumes the next consumer's weighted mixture of that prefix
and saved blocks, before output normalization. These are different tensors
when saved blocks exist.

`bind_aux(mixture=True)` compiles an explicit auxiliary-output variant for both
ordinary and Lamport-ring inputs. It uses the already computed score reduction
and softmax denominator to publish the BF16 mixture into a caller-owned,
potentially strided output slice. The default raw-prefix mode is unchanged.
Prefix, saved-block and normalized-output arithmetic remain unchanged; no
additional state owner, allocation in replay, or separate aggregation launch
is introduced.

The private component gate compares all five configured target taps against
stock aggregation and independent FP32 algebra, checks ordinary/ring inputs,
preserved outputs and guards, and replays graphs with changed inputs. Its
qualification applies only to those tested geometries; the payload digest
records the change but is not itself numerical or integration evidence.
