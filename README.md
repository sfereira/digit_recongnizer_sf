# Digit Recognizer Model

This is a Convolutional Neural Network (CNN) model trained to recognize handwritten digits. The model is built using the TensorFlow library and trained on the MNIST dataset.

## Libraries Used

* numpy
* pandas
* plotly.express
* matplotlib.pyplot
* sklearn
* tensorflow
* keras

## Data Processing

The model uses the MNIST dataset for training and testing. The dataset is loaded using the pandas library and pre-processed to normalize the data.

## Modeling

The model is built using the `TensorFlow` library and consists of multiple layers of 
`Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`, and `Dense layers`. The model is trained using the `Adam optimizer` and a learning rate of 0.001.

## Evaluation

The trained model is evaluated on the test dataset to validate the results. The evaluation metrics used are loss and accuracy.

## Conclusion

The model for digit recognition achieved a high level of accuracy on both the training and validation datasets. The model was able to correctly classify <b>99.9888%</b> of the training data and <b>99.5%</b> of the validation data.

The model was also evaluated on the test dataset, and achieved an accuracy of <b>99.2857%</b>. This indicates that the model is able to accurately classify digits that it has not seen before.

Overall, the model is effective for recognizing handwritten digits and can be used in various applications such as automated form processing, digitizing historical documents, and more.

