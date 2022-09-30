FROM python:3.10-slim-bullseye

RUN mkdir -p /home/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

#RUN apk update && apk add postgresql-dev gcc python3-dev musl-dev
RUN apt-get update

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . $APP_HOME

# chown all the files to the app user
#RUN chown -R 1001:1001 $APP_HOME

#USER 1001

CMD ["mlflow", "server", "-h", "0.0.0.0"]
