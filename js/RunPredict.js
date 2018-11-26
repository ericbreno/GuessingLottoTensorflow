const tf = require('@tensorflow/tfjs');
const { getSample, checkAccuracy } = require('./TrainingHelper');

const testPredict = (model, resultToTest) => {
    console.log('Predicting for:', resultToTest);
    // Use the model to do inference on a data point the model hasn't seen before:
    const [inputToPredict, expectedPredict] = getSample(resultToTest, resultToTest + 1);
    const predicted = model.predict(tf.tensor2d(inputToPredict, [1, 25]));

    const freqPropPredicted = predicted.dataSync();
    console.log(freqPropPredicted);

    // checking accuracy
    const guessed = expectedPredict[0].map((v, i) => v < freqPropPredicted[i] ? i + 1 : 0);
    checkAccuracy(guessed, resultToTest);
};

module.exports = {
    testPredict
};