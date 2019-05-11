// cleveland_bnn.js
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

    for (let k = 0; k < this.no; ++k) {
      this.oNodes[k] = U.logSig(oSums[k]);  // logistic sigmoid activation
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

  train(trainX, trainY, lrnRate, maxEpochs)
  {
    let hoGrads = U.matMake(this.nh, this.no, 0.0);
    let obGrads = U.vecMake(this.no, 0.0);
    let ihGrads = U.matMake(this.ni, this.nh, 0.0);
    let hbGrads = U.vecMake(this.nh, 0.0);

    let oSignals = U.vecMake(this.no, 0.0);
    let hSignals = U.vecMake(this.nh, 0.0);

    let n = trainX.length;  // 237
    let indices = U.arange(n);  // [0,1,..,236]
    let freq = Math.trunc(maxEpochs / 10);
    
    for (let epoch = 0; epoch < maxEpochs; ++epoch) {
      this.shuffle(indices);  //
      for (let ii = 0; ii < n; ++ii) {  // each item
        let idx = indices[ii];
        let X = trainX[idx];
        let Y = trainY[idx];
        this.eval(X);  // output stored in this.oNodes

        // compute output node signals
        for (let k = 0; k < this.no; ++k) {
          let derivative = (1 - this.oNodes[k]) * this.oNodes[k];  // logsig same as softmax!
          oSignals[k] = derivative * (this.oNodes[k] - Y[k]);  // E=(t-o)^2 
        }      

        // compute hidden-to-output weight gradients using output signals
        for (let j = 0; j < this.nh; ++j) {
          for (let k = 0; k < this.no; ++k) {
            hoGrads[j][k] = oSignals[k] * this.hNodes[j];
          }
        }

        // compute output node bias gradients using output signals
        for (let k = 0; k < this.no; ++k) {
          obGrads[k] = oSignals[k] * 1.0;  // 1.0 dummy input can be dropped
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

        // compute input-to-hidden weight gradients using hidden signals
        for (let i = 0; i < this.ni; ++i) {
          for (let j = 0; j < this.nh; ++j) {
            ihGrads[i][j] = hSignals[j] * this.iNodes[i];
          }
        }

        // compute hidden node bias gradients using hidden signals
        for (let j = 0; j < this.nh; ++j) {
          hbGrads[j] = hSignals[j] * 1.0;  // 1.0 dummy input can be dropped
        }

        // update input-to-hidden weights
        for (let i = 0; i < this.ni; ++i) {
          for (let j = 0; j < this.nh; ++j) {
            let delta = -1.0 * lrnRate * ihGrads[i][j];
            this.ihWeights[i][j] += delta;
          }
        }

        // update hidden node biases
        for (let j = 0; j < this.nh; ++j) {
          let delta = -1.0 * lrnRate * hbGrads[j];
          this.hBiases[j] += delta;
        }  

        // update hidden-to-output weights
        for (let j = 0; j < this.nh; ++j) {
          for (let k = 0; k < this.no; ++k) { 
            let delta = -1.0 * lrnRate * hoGrads[j][k];
            this.hoWeights[j][k] += delta;
          }
        }

        // update output node biases
        for (let k = 0; k < this.no; ++k) {
          let delta = -1.0 * lrnRate * obGrads[k];
          this.oBiases[k] += delta;
        }
      } // ii

      if (epoch % freq == 0) {
        let mse = this.meanSqErr(trainX, trainY).toFixed(4);
        let acc = this.accuracy(trainX, trainY).toFixed(4);

        let s1 = "epoch: " + epoch.toString();
        let s2 = " MSE = " + mse.toString();
        let s3 = " acc = " + acc.toString();

        console.log(s1 + s2 + s3);
      }
      
    } // epoch
  } // train()

  meanCrossEntErr(dataX, dataY)  // doesn't work for binary classification
  {
    let sumCEE = 0.0;  // sum of the cross entropy errors
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let Y = dataY[i];  // target output like (0, 1, 0)
      let oupt = this.eval(X);  // computed like (0.23, 0.66, 0.11)
      let idx = U.argmax(Y);  // find location of the 1 in target
      sumCEE += Math.log(oupt[idx]);
    }
    return -1 * sumCEE / dataX.length;
  }

  meanBinCrossEntErr(dataX, dataY)  // for binary problems
  {
    let sum = 0.0;
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let oupt = this.eval(dataX[i]);
      if (dataY[i] == 1) {  // target is 1
        sum += Math.log(oupt);
      }
      else {  // target is 0
        sum += Math.log(1.0 - oupt);
      }
    }
    return -1 * sum / dataX.length;
  }

  meanSqErr(dataX, dataY)
  {
    let sumSE = 0.0;
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let Y = dataY[i];  // target output 0 or 1
      let oupt = this.eval(X);  // computed like 0.2345
      for (let k = 0; k < this.no; ++k) {
        let err = Y[k] - oupt[k]  // target - computed
        sumSE += err * err;
      }
    }
    return sumSE / dataX.length;
  } 

  accuracy(dataX, dataY)
  {
    let nc = 0; let nw = 0;
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let Y = dataY[i];  // target output 0 or 1
      let oupt = this.eval(X);  // computed like 0.2345

      if (Y == 0 && oupt < 0.5 || Y == 1 && oupt >= 0.5) {
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
  console.log("\nBegin Cleveland Data binary classification demo  ");

  // 1. load data
  // raw data has 13 predictor values such as age, sex, BP, ECG 
  // normalized and encoded data has 18 values
  // value to predict: 0 = no heart disease, 1 = heart disease
  let trainX = U.loadTxt(".\\Data\\cleveland_train.txt", "\t", [0,1,2,3,4,
    5,6,7,8,9,10,11,12,13,14,15,16,17]);
  let trainY = U.loadTxt(".\\Data\\cleveland_train.txt", "\t", [18]);

  let testX = U.loadTxt(".\\Data\\cleveland_test.txt", "\t", [0,1,2,3,4,
    5,6,7,8,9,10,11,12,13,14,15,16,17]);
  let testY = U.loadTxt(".\\Data\\cleveland_test.txt", "\t", [18]);

  // 2. create network
  console.log("\nCreating an 18-100-1 tanh, logsig NN for Cleveland dataset");
  let seed = 0;
  let nn = new NeuralNet(18, 100, 1, seed);

  // 3. train network
  let lrnRate = 0.005;
  let maxEpochs = 600;
  console.log("\nStarting training with learning rate = 0.005 ");
  nn.train(trainX, trainY, lrnRate, maxEpochs);
  console.log("Training complete");

  // 4. evaluate model
  let trainAcc = nn.accuracy(trainX, trainY);
  let testAcc = nn.accuracy(testX, testY);
  console.log("\nAccuracy on training data = " +
    trainAcc.toFixed(4).toString()); 
  console.log("Accuracy on test data     = " +
    testAcc.toFixed(4).toString());

  // 5. save model
  let fn = ".\\Models\\cleveland_wts.txt";
  console.log("\nSaving model weights and biases to: ");
  console.log(fn);
  nn.saveWeights(fn);

  // 6. use trained model
  let unknownNorm = [0.7708, 1, -1, -1, -1, 0.2093, 0.1963, -1, -1, -1,
    0.4656, 1, 0.0161, 1, 0, 0.3333, 1, 0];
  let prediction = nn.eval(unknownNorm);

  console.log("\nNormalized features of patient to predict: ");
  U.vecShow(unknownNorm, 4, 6);
  console.log("\nPrediction (0 = no heart disease, 1 = heart disease): ");
  console.log(prediction[0].toFixed(4).toString());

  if (prediction[0] < 0.5) {
    console.log("patient is predicted to have low chance of heart disease");
  }
  else {
    console.log("patient is predicted to have high chance of heart disease");
  }

  process.stdout.write("\033[0m");  // reset
  console.log("\nEnd demo");
} // main()

main();
