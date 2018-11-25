const tf = require('@tensorflow/tfjs');
const { BuildModel } = require('./js/BuildModel');
const { buildTrainingSample, getSample, checkAccuracy } = require('./js/TrainingHelper');

require('@tensorflow/tfjs-node');

const firstRes = 1000,
    lastRes = 1500,
    epochs = 1000,
    stepsPerEpoch = 1,
    layerWidth = 25,
    inputWidth = 25,
    layers = 500;

const model = BuildModel(layerWidth, inputWidth, layers);
const [inputs, outputs] = buildTrainingSample(firstRes, lastRes);

// Train the model using the data.
model.fit(inputs, outputs, { epochs, stepsPerEpoch }).then(() => {
    const resultToPredict = lastRes + 1;
    console.log('Predicting for:', resultToPredict);
    // Use the model to do inference on a data point the model hasn't seen before:
    const [inputToPredict, expectedPredict] = getSample(resultToPredict, resultToPredict + 1);
    const predicted = model.predict(tf.tensor2d(inputToPredict, [1, 25]));

    model.save('file://models/last_run');

    const freqPropPredicted = predicted.dataSync();
    console.log(freqPropPredicted);

    // checking accuracy
    const guessed = expectedPredict[0].map((v, i) => v < freqPropPredicted[i] ? i + 1 : 0);
    checkAccuracy(guessed, resultToPredict);
});

