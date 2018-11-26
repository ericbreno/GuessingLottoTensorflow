# Neural Network Lotto Guesser, maybe
Just learning about Artificial Intelligence with [Google TensorflowJS](https://js.tensorflow.org/).  
Training a model with lotto historical data and guessing results, just for fun.

Ps: the data is already pre-processed by another project and grouped, etc. Maybe it's
format seems weird, but somehow it makes sense.

### How to test a model
```
node testModel.js {path_to_saved_model_directory} {resultToTestPredict}
```  
Example (working):  
```
node testModel.js models/run_5k_500_layers/ 1500
```  
No guarantees if parameters are invalid ¯\\\_(ツ)\_/¯  
Current models were generated with "run_{epochs}_{layers}_layers", training with sample 400-1500.

### How to create your own model
**Important:** Every new model is saved under directory `models/last_run` (_git-ignored_ purposely).

#### Fast setup (minimum)
Change variables in `createModel.js` and _voilá_ (run it):
```js
const
    // Training sample limits, cannot be over 1500 neither under 2 (at least 5 recommended)
    firstRes = 1200,
    lastRes = 1400,

    // Model core, number of middle layers added
    layers = 50,

    // Model params. Actually, should not change this, unless you're changing the 
    // analysis sample and everything else. Our lotto has only 25 numbers (1-25).
    layerWidth = 25,
    inputWidth = 25,

    // Model fit. Training loops, basically.
    epochs = 150,
    stepsPerEpoch = 1;
```

Once it's done everything is auto-saved in mentioned directory.

#### I really want to test other things
A good start is modifying `js/BuildModel.js`, there you can update how the model works as well
as the layers and it's configuration.  
Other change point is `js/TrainingHelper.js`, there you can update how the training sample is
extracted from analysed data (default is sequential results, note that the data is incremental). 
Maybe adding some random (or not so random) picked results. Also, the accuracy-checker is in there, 
feel free if you want to improve it's logger.


