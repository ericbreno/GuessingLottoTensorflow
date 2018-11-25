const tf = require('@tensorflow/tfjs');

const BuildModel = (layerWidth = 25, inputWidth = 25, layers = 100) => {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: inputWidth, // first and last must have input size as width
        inputShape: [inputWidth]
    }));

    for (let i = 0; i < layers; i++) {
        model.add(tf.layers.dense({
            units: layerWidth,
            inputShape: [inputWidth]
        }));
    }

    model.add(tf.layers.dense({
        units: inputWidth, // first and last must have input size as width
        inputShape: [inputWidth]
    }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    return model;
}

module.exports.BuildModel = BuildModel;