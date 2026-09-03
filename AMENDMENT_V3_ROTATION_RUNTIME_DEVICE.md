# V3 rotation runtime device amendment

Date: 2026-09-03. This amendment was made after conditional-fit stock and H16 completed, after learned failed during server construction, and before any selection-wave target was opened.

The learned endpoint failed closed before inference because vLLM constructed its rank-local learned rotation on CUDA while the runtime orthogonality check constructed its identity reference on CPU. The exact exception was `Expected all tensors to be on the same device, but found at least two devices, cuda:2 and cpu`. No learned KLD row and no selection row was produced.

The correction creates the identity reference with the learned rotation's dtype and device. It changes neither the learned rotation values nor the input transform, weight transform, quantized checkpoint, role manifest, KLD implementation, statistical rule, candidate set, nor H16 behavior. The old freeze receipt remains immutable. A new freeze receipt hashes the corrected runtime and clean implementation commit before learned conditional-fit resumes and before selection wave 1 opens.

The completed H16 conditional-fit row remains valid as a failed endpoint under the prior runtime because the device-only correction does not alter H16 construction or arithmetic. Its full 16-window mean KLD was 2.8084217364923516 versus stock 0.0531151514327003. This is retained as evidence of a functional mismatch for the fixed H16 checkpoint/runtime pair, not interpreted as evidence against the abstract Hadamard method.
