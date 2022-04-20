FROM python:3.8-slim

RUN apt-get update

ENV APP_HOME main/
WORKDIR $APP_HOME

COPY . .

RUN mkdir data

RUN pip install -r requirements_train.txt

ENTRYPOINT ["python3", "full_pipeline.py"]