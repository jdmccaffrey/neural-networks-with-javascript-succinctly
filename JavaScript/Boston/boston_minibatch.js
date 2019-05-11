// boston_minibatch.js
// ES6

let U = require("../Utilities/utilities_lib.js");
let FS = require("fs");

// =============================================================================

class NeuralNet
{
  constructor(numInput, numHidden, numOutput, seed)
  {
    this.rnd = new U.Erratic(seed);

    this.ni = numInput; 
    this.nh = numHidden;
    this.no = numOutput;

    this.iNodes = U.vecMake(this.ni, 0.0);
    this.hNodes = U.vecMake(this.nh, 0.0);
    this.oNodes = U.vecMake(this.no, 0.0);

    this.ihWeights = U.matMake(this.ni, this.nh, 0.0);
    this.hoWeights = U.matMake(this.nh, this.no, 0.0);

    this.hBiases = U.vecMake(this.nh, 0.0);
    this.oBiases = U.vecMake(this.no, 0.0);

    this.initWeights();
  }

  initWeights()
  {
    let lo = -0.01;
    let hi = 0.01;
    for (let i = 0; i < this.ni; ++i) {
      for (let j = 0; j < this.nh; ++j) {
        this.ihWeights[i][j] = (hi - lo) * this.rnd.next() + lo;
      }
    }

    for (let j = 0; j < this.nh; ++j) {
      for (let k = 0; k < this.no; ++k) {
        this.hoWeights[j][k] = (hi - lo) * this.rnd.next() + lo;
      }
    }
  } 

  eval(X)
  {
    // regresion: no output activation
    let hSums = U.vecMake(this.nh, 0.0);
    let oSums = U.vecMake(this.no, 0.0);
    
    this.iNodes = X;

    for (let j = 0; j < this.nh; ++j) {
      for (let i = 0; i < this.ni; ++i) {
        hSums[j] += this.iNodes[i] * this.ihWeights[i][j];
      }
      hSums[j] += this.hBiases[j];
      this.hNodes[j] = U.hyperTan(hSums[j]);
    }

    for (let k = 0; k < this.no; ++k) {
      for (let j = 0; j < this.nh; ++j) {
        oSums[k] += this.hNodes[j] * this.hoWeights[j][k];
      }
      oSums[k] += this.oBiases[k];
    }

    // this.oNodes = U.softmax(oSums);  // aka "Identity"
    for (let k = 0; k < this.no; ++k) {
      this.oNodes[k] = oSums[k];
    }

    let result = [];
    for (let k = 0; k < this.no; ++k) {
      result[k] = this.oNodes[k];
    }
    return result;
  } // eval()

  setWeights(wts)
  {
    // order: ihWts, hBiases, hoWts, oBiases
    let p = 0;

    for (let i = 0; i < this.ni; ++i) {
      for (let j = 0; j < this.nh; ++j) {
        this.ihWeights[i][j] = wts[p++];
      }
    }

    for (let j = 0; j < this.nh; ++j) {
      this.hBiases[j] = wts[p++];
    }

    for (let j = 0; j < this.nh; ++j) {
      for (let k = 0; k < this.no; ++k) {
        this.hoWeights[j][k] = wts[p++];
      }
    }

    for (let k = 0; k < this.no; ++k) {
      this.oBiases[k] = wts[p++];
    }
  } // setWeights()

  getWeights()
  {
    // order: ihWts, hBiases, hoWts, oBiases
    let numWts = (this.ni * this.nh) + this.nh +
      (this.nh * this.no) + this.no;
    let result = U.vecMake(numWts, 0.0);
    let p = 0;
    for (let i = 0; i < this.ni; ++i) {
      for (let j = 0; j < this.nh; ++j) {
        result[p++] = this.ihWeights[i][j];
      }
    }

    for (let j = 0; j < this.nh; ++j) {
      result[p++] = this.hBiases[j];
    }

    for (let j = 0; j < this.nh; ++j) {
      for (let k = 0; k < this.no; ++k) {
        result[p++] = this.hoWeights[j][k];
      }
    }

    for (let k = 0; k < this.no; ++k) {
      result[p++] = this.oBiases[k];
    }
    return result;
  } // getWeights()

  shuffle(v)
  {
    // Fisher-Yates
    let n = v.length;
    for (let i = 0; i < n; ++i) {
      let r = this.rnd.nextInt(i, n);
      let tmp = v[r];
      v[r] = v[i];
      v[i] = tmp;
    }
  }

  miniTrain(trainX, trainY, batchSize, lrnRate, maxEpochs)
  {
    let N = trainX.length;
    let indices = U.arange(N);  // [0,1,2, . .]
    this.shuffle(indices);
    let ptr = 0;  // into indices
    let batchIndices = U.vecMake(batchSize, 0);
    let freq = Math.trunc(maxEpochs / 10);

    for (let epoch = 0; epoch < maxEpochs; ++epoch) {

      if (ptr + batchSize > N) {  // no more batch left
        this.shuffle(indices);
        ptr = 0;
      }
      else {  // get next batch and train using it
        for (let j = 0; j < batchSize; ++j) {
          batchIndices[j] = indices[ptr++];
        }
        this.processBatch(trainX, trainY, batchIndices, lrnRate, maxEpochs);
      }
      
      if (epoch % freq == 0) {
        let mse = this.meanSqErr(trainX, trainY).toFixed(4);
        let acc = this.accuracy(trainX, trainY, 0.15).toFixed(4);

        let s1 = "epoch: " + epoch.toString();
        let s2 = " MSE = " + mse.toString();
        let s3 = " acc = " + acc.toString();

        console.log(s1 + s2 + s3);
      }
    } // each epoch

  } // train

  processBatch(trainX, trainY, batchIndices, lrnRate, maxEpochs)
  {
    let batchSize = batchIndices.length;
    let oSignals = U.vecMake(this.no, 0.0);
    let hSignals = U.vecMake(this.nh, 0.0);
    let ihWtsAccGrads = U.matMake(this.ni, this.nh, 0.0);  // accumulated grads
    let hBiasAccGrads = U.vecMake(this.nh, 0.0);
    let hoWtsAccGrads = U.matMake(this.nh, this.no, 0.0);
    let oBiasAccGrads = U.vecMake(this.no, 0.0);

    for (let ii = 0; ii < batchSize; ++ii) {  // each item in batch
      let idx = batchIndices[ii];
      let X = trainX[idx];
      let Y = trainY[idx];
      this.eval(X);  // output stored in this.oNodes
 
      // compute output node signals
      for (let k = 0; k < this.no; ++k) {
        let derivative = 1;  // identity activation for regression
        oSignals[k] = derivative * (this.oNodes[k] - Y[k]);  // E=(t-o)^2 
      }      

      // compute and accumulate hidden-to-output weight gradients
      for (let j = 0; j < this.nh; ++j) {
        for (let k = 0; k < this.no; ++k) {
          let grad = oSignals[k] * this.hNodes[j];
          hoWtsAccGrads[j][k] += grad;
        }
      } 

      // compute and accumulate output node bias gradients
      for (let k = 0; k < this.no; ++k) {
        let grad = oSignals[k] * 1.0;  // 1.0 dummy input can be dropped
        oBiasAccGrads[k] += grad;
      }

      // compute hidden node signals
      for (let j = 0; j < this.nh; ++j) {
        let sum = 0.0;
        for (let k = 0; k < this.no; ++k) {
          sum += oSignals[k] * this.hoWeights[j][k];
        }
        let derivative = (1 - this.hNodes[j]) * (1 + this.hNodes[j]);  // tanh
        hSignals[j] = derivative * sum;
      } 

      // compute and accumulate input-to-hidden weight gradients
      for (let i = 0; i < this.ni; ++i) {
        for (let j = 0; j < this.nh; ++j) {
          let grad = hSignals[j] * this.iNodes[i];
          ihWtsAccGrads[i][j] += grad;
        }
      }

      // compute and accumulate hidden node bias gradients
      for (let j = 0; j < this.nh; ++j) {
        let grad = hSignals[j] * 1.0;  // 1.0 dummy input can be dropped
        hBiasAccGrads[j] += grad;
      }

    } // each item in batch  

    // after all gradients accumulated, update using grad average

    let n = batchSize;

    // update input-to-hidden weights
    for (let i = 0; i < this.ni; ++i) {
      for (let j = 0; j < this.nh; ++j) {
        let delta = -1.0 * lrnRate * (ihWtsAccGrads[i][j] / n);
        this.ihWeights[i][j] += delta;
      }
    }

    // update hidden node biases
    for (let j = 0; j < this.nh; ++j) {
      let delta = -1.0 * lrnRate * (hBiasAccGrads[j] / n);
      this.hBiases[j] += delta;
    }  

    // update hidden-to-output weights
    for (let j = 0; j < this.nh; ++j) {
      for (let k = 0; k < this.no; ++k) { 
        let delta = -1.0 * lrnRate * (hoWtsAccGrads[j][k] / n);
        this.hoWeights[j][k] += delta;
      }
    }

    // update output node biases
    for (let k = 0; k < this.no; ++k) {
      let delta = -1.0 * lrnRate * (oBiasAccGrads[k] / n);
      this.oBiases[k] += delta;
    }   
  } // processBatch

  // cross entropy error not applicable to regression problems

  meanSqErr(dataX, dataY)
  {
    let sumSE = 0.0;
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let y = dataY[i];  // target output like [2.3] as matrix
      let oupt = this.eval(X);  // computed like [2.07]
      let err = y[0] - oupt[0];
      sumSE += err * err;
    }
    return sumSE / dataX.length;
  } 

  accuracy(dataX, dataY, pctClose)
  {
    // correct if predicted is within pct of target
    let nc = 0; let nw = 0;
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let y = dataY[i];  // target output 
      let oupt = this.eval(X);  // computed output

      if (Math.abs(oupt[0] - y[0]) < Math.abs(pctClose * y[0])) {
        ++nc;
      }
      else {
        ++nw;
      }
    }
    return nc / (nc + nw);
  }

  saveWeights(fn)
  {
    let wts = this.getWeights();
    let n = wts.length;
    let s = "";
    for (let i = 0; i < n-1; ++i) {
      s += wts[i].toString() + ",";
    }
    s += wts[n-1];

    FS.writeFileSync(fn, s);
  }

  loadWeights(fn)
  {
    let n = (this.ni * this.nh) + this.nh + (this.nh * this.no) + this.no;
    let wts = U.vecMake(n, 0.0);
    let all = FS.readFileSync(fn, "utf8");
    let strVals = all.split(",");
    let nn = strVals.length;
    if (n != nn) {
      throw("Size error in NeuralNet.loadWeights()");
    }
    for (let i = 0; i < n; ++i) {
      wts[i] = parseFloat(strVals[i]);
    }
    this.setWeights(wts);
  }


} // NeuralNet

// =============================================================================

function main()
{
  process.stdout.write("\033[0m");  // reset
  process.stdout.write("\x1b[1m" + "\x1b[37m");  // bright white
  console.log("\nBegin Boston Data regression using batch training ");

  // 1. load data
  let trainX = U.loadTxt(".\\Data\\boston_train.txt", "\t",
    [0,1,2,3,4,5,6,7,8,9,10,11,12]);
  let trainY = U.loadTxt(".\\Data\\boston_train.txt", "\t", [13]);
  let testX = U.loadTxt(".\\Data\\boston_test.txt", "\t",
    [0,1,2,3,4,5,6,7,8,9,10,11,12]);
  let testY = U.loadTxt(".\\Data\\boston_test.txt", "\t", [13]);

  // 2. create network
  console.log("\nCreating a 13-100-1 tanh, Identity NN for Boston dataset");
  let seed = 0;
  let nn = new NeuralNet(13, 100, 1, seed);

  // 3. train network
  let lrnRate = 0.10;
  let maxEpochs = 50000;
  let batchSize = 16;
  console.log("\nStarting minibatch training with learn rate = 0.10 batch size = 16");
  nn.miniTrain(trainX, trainY, batchSize, lrnRate, maxEpochs);
  console.log("Training complete");

  // 4. evaluate model
  let trainAcc = nn.accuracy(trainX, trainY, 0.15);
  let testAcc = nn.accuracy(testX, testY, 0.15);
  console.log("\nAccuracy on training data = " +
    trainAcc.toFixed(4).toString()); 
  console.log("Accuracy on test data     = " +
    testAcc.toFixed(4).toString());

  // 5. save model
  let fn = ".\\Models\\boston_wts.txt";
  console.log("\nSaving model weights and biases to: ");
  console.log(fn);
  nn.saveWeights(fn);

  // 6. use trained model
  let unknownRaw = [0.04819, 80, 3.64, 0, 0.392, 6.108, 32, 9.2203, 1, 315,
    16.4, 392.89, 6.57];
  let unknownNorm = [0.000471, 0.800000, 0.116569, -1, 0.014403, 0.488025,
    0.299691, 0.735726, 0.000000, 0.244275, 0.404255, 0.989889, 0.133554];
  let predicted = nn.eval(unknownNorm);

  console.log("\nRaw features of town to predict: ");
  U.vecShow(unknownRaw, 4, 7);

  console.log("\nNormalized features of town to predict: ");
  U.vecShow(unknownNorm, 4, 7);

  console.log("\nPredicted median house price in town: ");
  console.log(predicted[0].toFixed(6).toString());

  let predPrice = predicted[0] * 10000;
  console.log("( $" + predPrice.toFixed(2).toString() + " )" );
  //console.log(predPrice);
  //let politics = ["conservative", "moderate", "liberal"];
  //let predIdx = U.argmax(predicted);
  //let predPolitic = politics[predIdx];
  //console.log("Predicted politic = " + predPolitic);

  process.stdout.write("\033[0m");  // reset
  console.log("\nEnd demo");
} // main()

main();


