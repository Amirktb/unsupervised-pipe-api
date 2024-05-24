FROM python:3.11

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /usr/src/unsupervised-pipe-api

Add ./unsupervised-ml-api /usr/src/unsupervised-pipe-api/
RUN pip install --upgrade pip
RUN pip install -r /usr/src/unsupervised-pipe-api/requirements.txt

RUN chmod +x /usr/src/unsupervised-pipe-api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]
