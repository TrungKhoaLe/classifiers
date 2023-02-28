# :blossom: Flower Classifier

This project leverages deep learning algorithms to classify different kinds of
flowers with high accuracy.

This project assumes familiarity with Pytorch, Pytorch Lightning, Weights and
Biases (W&B), and Gradio, but some basic commands are available to get you started.

## Setting up the Python environment

```
pipenv install --ignore-pipfile
```

## Using the pretrained model

```
python training/stage_model.py --fetch --entity=khoale --from_project=flower_classification
```


## Fine-tuning a deep learning model on the flower dataset

* If you want to stream trainging logs and upload model artifacts to W&B use the
  flowwing command, notice the flag --wand in the command.

```
python training/run_experiment.py --max_epochs=8 --gpus='0,' \
--num_workers=24 --model_class=VGG16Classifier --data_class=Flowers \
--fc1_dim=8192 --fc2_dim=2048 --batch_size=32 --wandb
```

* If you don't want to touch W&B, simply remove the --wandb out of the command

```
python training/run_experiment.py --max_epochs=8 --gpus='0,' \
--num_workers=24 --model_class=VGG16Classifier --data_class=Flowers \
--fc1_dim=8192 --fc2_dim=2048 --batch_size=32
```

In both cases, you can mess around hyperparameters, i.e. fc1\_dim, fc2\_dim, or
get rid of the --gpus flag if you don't have ones.


## Serializing the trained model

```
python training/stage_model.py --entity='your_account_name'
```

*Side note: At the moment, support model staging from W&B only*

## Running the gradio app

If the pretrained model has been downloaded and lived in the right directory,
you can run the following command to enjoy the final product.

```
python flower_classifier/app_gradio/app.py
```

## Future work

* Add the training from scratch switch

* Allow to stage model locally
