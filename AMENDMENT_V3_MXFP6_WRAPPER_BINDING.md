# V3 MXFP6 scale-geometry wrapper binding correction

Date: 2026-09-03. The first attempt to attach the virtual-TP scale geometry failed during model construction, before weight loading, readiness, or inference. Python functions assigned to an instance are not descriptor-bound, while vLLM calls `create_weights(layer=...)`; the wrapper incorrectly required a positional `target` argument.

The wrapper now transparently forwards arbitrary positional and keyword arguments to the original bound method and obtains the routed-experts object from the `layer` keyword (with a positional fallback). The intended group-size attributes, tensor artifacts, numerical methods, experiment roles, and decision rules are otherwise unchanged. The failed startup log remains preserved and the identical canary is repeated before bulk work.
