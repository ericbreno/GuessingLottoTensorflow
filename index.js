const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const data = require('./data/analysis.json');
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

const init = 100, end = 1600;
const [x, y] = getSample(init, end);

const model = tf.sequential();
model.add(tf.layers.dense({
    units: 25,
    inputShape: [25]
}));

model.add(tf.layers.dense({
    units: 25,
    inputShape: [25]
}));

model.add(tf.layers.dense({
    units: 25,
    inputShape: [25]
}));

model.add(tf.layers.dense({
    units: 25,
    inputShape: [25]
}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
});

// Generate some synthetic data for training.
const xs = tf.tensor2d(x, [end - init, 25]);
const ys = tf.tensor2d(y, [end - init, 25]);

// Train the model using the data.
model.fit(xs, ys, { epochs: 5000 }).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    const [xx, yy] = getSample(end + 1, end + 2);
    const s = model.predict(tf.tensor2d(xx, [1, 25]));

    model.save('file://models/sec_5k');

    const res = s.dataSync();
    console.log(res);

    const [xxg, yyg] = getSample(end, end + 1);
    const guessed = yyg[0].map((v, i) => v < res[i] ? i + 1 : 0);
    console.log('Guessed:', guessed.filter(k => k).join('-'));
    console.log('Real:   ', data[end].nextRes.sort((a, b) => a - b).join('-'));
});

