FROM python:3.8-slim

RUN apt-get update

ENV APP_HOME main/
WORKDIR $APP_HOME

COPY req/requirements_train_cuda102.txt .
COPY *.py ./

RUN mkdir data

RUN pip install -r requirements_train_cuda102.txt

ENTRYPOINT ["python3", "full_pipeline.py"]