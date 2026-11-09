FROM python:3.11.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends bash procps util-linux && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 sympy==1.13.1
RUN python -m venv /opt/private-grader
RUN /opt/private-grader/bin/pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5
RUN chmod -R go-rwx /opt/private-grader
RUN mkdir -p /root /logs/verifier /root/dominating_set_submission/runtime_outputs/public /root/dominating_set_submission/runtime_outputs/heldout
COPY harness_run.py /harness_run.py
COPY . /tests/
RUN chmod 555 /harness_run.py && chmod 700 /tests /logs/verifier && chmod 755 /tests/test.sh
WORKDIR /root
