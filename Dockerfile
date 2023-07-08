FROM python:3.11

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501
ENTRYPOINT [ "streamlit", "run", "deploy.py" ]