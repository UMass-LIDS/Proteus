FROM python:3.9.6

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]
