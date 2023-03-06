# :blossom: Flower Classifier

This project leverages deep learning algorithms to classify different kinds of
flowers with high accuracy.

This project assumes familiarity with Pytorch, Pytorch Lightning, Weights and
Biases (W&B), and Gradio, but some basic commands are available to get you started.

## Prerequisite

This project uses W&B to keep track of experiments. Thus, in order to proceed further
steps, a W&B account is required.

## Setting up the Python environment

```
pipenv install --ignore-pipfile
```

## Using the pretrained model

```
python training/stage_model.py --fetch --entity=khoale --from_project=flower_classification
```


## Fine-tuning a deep learning model on the flower dataset

```
python training/run_experiment.py --max_epochs=8 --gpus='0,' \
--num_workers=24 --model_class=VGG16Classifier --data_class=Flowers \
--fc1_dim=8192 --fc2_dim=2048 --batch_size=32 --wandb
```

Feel free to change the values of fc1\_dim, fc2\_dim, or to
get rid of the --gpus flag if you don't have ones.


## Serializing the trained model

```
python training/stage_model.py --entity='your_account_name'
```

## Running the gradio app

If the pretrained model has been downloaded and lived in the right directory,
you can run the following command to enjoy the final product.

```
python flower_classifier/app_gradio/app.py
```

## Deploying to AWS

* Build a container image from the provided
  [Dockerfile](api_serverless/Dockerfile)
* Upload the container image to the [Elastic Container Registry (ECR)](https://aws.amazon.com/ecr/)
* Create a Lambda function
* Add an HTTP endpoint with a URL
* Connect to the frontend

