FROM public.ecr.aws/lambda/python:3.8

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip==22.0.4
RUN pip install -r requirements.txt

COPY flower_classifier/ ./flower_classifier
COPY api_serverless/api.py ./api.py

CMD ["api.handler"]
