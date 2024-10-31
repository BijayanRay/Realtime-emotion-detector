# realtime-emotion-detector

This is an real time emotion detector. One can clone the repository and do the following to use it. 

To use the trained model (trained on this [dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)):
- Execute ``pip install -r requirements_trained_model.txt``
- Go to the ``src`` directory.
- Execute ``python app.py``.
- Quit the app window by pressing ``q``.

To train the model manually and use that:
- Execute ``pip install -r requirements_training.txt``
- Open the notebook on kaggle and use the [dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer) or some other dataset for training.
- Download the model and save it in the same directory as the ``app.py`` file. 
- Execute ``python app.py``.

The ChatGPT (free version) has been used in developing the project and PyTorch libraries have been used in building and training the neural network. 
