# realtime-emotion-detector

This is an real time emotion detector. One can clone the repository and do the following to use it. 

To get a demo online that uses the trained model (trained on this [dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)):
- Go to https://realtime-emotion-detector.streamlit.app/.
- Note that the online demo does not have the real-time update of emotion yet, it captures the photo and then outputs the emotion. However the offline ``src/app.py`` does (almost) the real-time update of emotion of the person seen through the webcam.

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
