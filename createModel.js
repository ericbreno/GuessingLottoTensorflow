const { BuildModel } = require('./js/BuildModel');
const { buildTrainingSample } = require('./js/TrainingHelper');
const { testPredict } = require('./js/RunPredict');

require('@tensorflow/tfjs-node');

const
    // Training sample limits, cannot be over 1832 neither under 2 (at least 5 recommended)
    firstRes = 1500,
    lastRes = 1832,

    // Model core, number of middle layers added
    layers = 50,

    // Model params. Actually, should not change this, unless you're changing the 
    // analysis sample and everything else. Our lotto has only 25 numbers (1-25).
    layerWidth = 25,
    inputWidth = 25,

    // Model fit. Training loops, basically.
    epochs = 200,
    stepsPerEpoch = 20;

const model = BuildModel(layerWidth, inputWidth, layers);
const [inputs, outputs] = buildTrainingSample(firstRes, lastRes);

// Train the model using the data.
model.fit(inputs, outputs, { epochs, stepsPerEpoch }).then(() => {
    model.save('file://models/last_run');

    testPredict(model, lastRes + 1);
});
