const tf = require('@tensorflow/tfjs');
const { BuildModel } = require('./js/BuildModel');
const { buildTrainingSample, getSample, checkAccuracy } = require('./js/TrainingHelper');

require('@tensorflow/tfjs-node');

const init = 400, end = 1500, epochs = 50, stepsPerEpoch = 10;

const model = BuildModel();
const [xs, ys] = buildTrainingSample(init, end);

// Train the model using the data.
model.fit(xs, ys, { epochs, stepsPerEpoch }).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    const [xx, yy] = getSample(end + 1, end + 2);
    const predicted = model.predict(tf.tensor2d(xx, [1, 25]));

    model.save('file://models/last_run');

    const freqPropPredicted = predicted.dataSync();
    console.log(freqPropPredicted);

    // checking accuracy
    const [xxg, yyg] = getSample(end, end + 1);
    const guessed = yyg[0].map((v, i) => v < freqPropPredicted[i] ? i + 1 : 0);
    checkAccuracy(guessed, end);
});

