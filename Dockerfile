FROM python:3.10-slim

WORKDIR /code 

RUN mkdir -p /code/pages
RUN mkdir -p /images
COPY ./images/benign.JPEG /images/benign.JPEG
COPY ./images/malignant.JPEG /images/malignant.JPEG
COPY ./skin_cancer_detector.py  /code/skin_cancer_detector.py
COPY ./pages/about.py /code/pages/about.py
COPY ./requirements.txt /code/requirements.txt
COPY ./model.keras /code/model.keras
RUN pip install -r requirements.txt


EXPOSE 80

CMD ["streamlit", "run", "skinscan.py", "--server.port", "80"]

