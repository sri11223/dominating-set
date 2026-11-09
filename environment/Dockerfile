FROM python:3.11.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends bash procps util-linux && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 sympy==1.13.1
WORKDIR /root
RUN mkdir -p /root/dominating_set_submission/runtime_outputs/public /root/dominating_set_submission/runtime_outputs/heldout
COPY data /root/data
