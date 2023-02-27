# :blossom: Flower Classifier

This project leverages deep learning algorithms to classify different kinds of
flowers with high accuracy.

This project assumes familiarity with Pytorch, Pytorch Lightning, Weights and
Biases (W&B), and Gradio, but some basic commands are available to get you started.

## Fine-tuning a deep learning model on the flower dataset

* If you want to stream trainging logs and upload model artifacts to W&B use the
  flowwing command, notice the flag --wand in the command.
```
python training/run\_experiment.py --max\_epochs=8 --gpus='0,' \
--num\_workers=24 --model\_class=VGG16Classifier --data\_class=Flowers \
--fc1\_dim=8192 --fc2\_dim=2048 --wandb --batch\_size=32
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
python training/stage\_model.py --entity=DEFAULT
```

*Side note: At the moment, support model staging from W&B only*

## Running the gradio app

```
python flower_classifier/app_gradio/app.py
```

## Future work

* Add the training from scratch switch

* Allow to stage model locally
