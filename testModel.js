const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const { testPredict } = require('./js/RunPredict');

const modelToTest = process.argv[2];
const resultToTest = Number(process.argv[3]);
console.log('Testing for result: ', resultToTest);
console.log('Testing model:      ', modelToTest);

const run = async () => {
    const model = await tf.loadModel('file://' + modelToTest + 'model.json');
    testPredict(model, resultToTest);
};

run();
