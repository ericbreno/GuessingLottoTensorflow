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

    const xs = tf.tensor2d(x, [end - init, 25]);
    const ys = tf.tensor2d(y, [end - init, 25]);
    return [xs, ys];
};

const checkAccuracy = (guessed, lastTrained) => {
    console.log('Guessed:', guessed.filter(k => k).join('-'));
    console.log('Real:   ', data[lastTrained].nextRes.sort((a, b) => a - b).join('-'));
};

module.exports = {
    buildTrainingSample,
    getSample,
    checkAccuracy
};