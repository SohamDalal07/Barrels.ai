FROM python:3.8.10

#copy the following files and folders
COPY ./app /app
COPY ./requirements.txt /requirements.txt
COPY ./pipelines /pipelines
COPY .env .env

RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
#install whatever is necessary
RUN pip install --upgrade pip
RUN python3 -m pip --no-cache-dir install -r requirements.txt
# The project originally downloaded the pretrained YOLO model during image build
# using a pypyr pipeline. That requires S3 credentials during build which is
# inconvenient. We skip automatic download here — instead, pre-download the
# file and place it at ./models/yolov3/best_weights_final_18.hdf5 on the host
# and mount ./models into the container at runtime.
# To manually download the model before building/running, you can either:
#  - run the pypyr pipeline locally (requires AWS/DO credentials in a .env file):
#      python3 -m pypyr pipelines/ai-model-download
#  - or use the AWS CLI / curl to copy the file into ./models/yolov3/
# If you prefer automatic download during image build, uncomment the line below.
# RUN python3 -m pypyr /pipelines/ai-model-download

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--reload"]