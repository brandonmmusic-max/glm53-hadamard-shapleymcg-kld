# Supply the immutable verdictai base reference recorded in image-record.json.
ARG BASE_IMAGE=verdictai/trellismx:glm53-flash-p8-base-v11-20260907@sha256:9846694e7dd3546f6ef3d259ecd2c129a5d54147662a76d32c5a3d15673f5e25
FROM ${BASE_IMAGE}
COPY rootfs/ /
ENV PYTHONPATH=/opt/p8-mtp-bootstrap:/opt/p8-coupled-runtime:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x
RUN /opt/venv/bin/python /opt/p8-coupled-runtime/verify_release.py
LABEL org.opencontainers.image.title="TrellisMX P8 GLM-5.3-Flash RC5 runtime" \
      org.opencontainers.image.licenses="LicenseRef-ShapleyMCG-1.0" \
      org.klc.inference-only="true" \
      org.klc.qualification="release-candidate-not-final"
