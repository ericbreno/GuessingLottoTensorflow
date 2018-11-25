const tf = require('@tensorflow/tfjs');
const data = require('../data/analysis.json');

const getSample = (init, end) => {
    const slice = data.slice(init, end);
    return slice.reduce(([value, result], { freqProp, nextFreqProp }) => {
        const v = [], r = [];
        for (let i = 1; i < 26; i++) {
            v.push(freqProp[i]);
            r.push(nextFreqProp[i])
        }
        value.push(v);
        result.push(r);
        return [value, result];
    }, [[], []]);
};

const buildTrainingSample = (init = 100, end = 1600) => {
    const [x, y] = getSample(init, end);

    const inputs = tf.tensor2d(x, [end - init, 25]);
    const outputs = tf.tensor2d(y, [end - init, 25]);
    return [inputs, outputs];
};

const checkAccuracy = (guessed, predicted) => {
    console.log('Checking for result:', predicted);

    const correctNumbers = data[predicted - 1].nextRes;
    const guessedNumbers = guessed.filter(k => k);
    const guessedRight = guessedNumbers.filter(n => correctNumbers.includes(n));

    console.log('Guessed:', guessedNumbers.join('-'));
    //                           previous.nextRes
    console.log('Real:   ', correctNumbers.sort((a, b) => a - b).join('-'));
    console.log('Ratio:  ', `${guessedRight.length}/15 from ${guessedNumbers.length} guessed.`);
};

module.exports = {
    buildTrainingSample,
    getSample,
    checkAccuracy
};