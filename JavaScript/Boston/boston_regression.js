// boston_regression.js
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

    // this.oNodes = U.softmax(oSums);
    for (let k = 0; k < this.no; ++k) {  // aka "Identity"
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

  train(trainX, trainY, lrnRate, maxEpochs)
  {
    // regression: no output activation => f(x)=x => f'(x)=1 
    let hoGrads = U.matMake(this.nh, this.no, 0.0);
    let obGrads = U.vecMake(this.no, 0.0);
    let ihGrads = U.matMake(this.ni, this.nh, 0.0);
    let hbGrads = U.vecMake(this.nh, 0.0);

    let oSignals = U.vecMake(this.no, 0.0);
    let hSignals = U.vecMake(this.nh, 0.0);

    let n = trainX.length;  // 406
    let indices = U.arange(n);  // [0,1,..,405]
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
          // let derivative = (1 - this.oNodes[k]) * this.oNodes[k];  // softmax
          let derivative = 1;  // identity activation
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
        let acc = this.accuracy(trainX, trainY, 0.15).toFixed(4);

        let s1 = "epoch: " + epoch.toString();
        let s2 = " MSE = " + mse.toString();
        let s3 = " acc = " + acc.toString();

        console.log(s1 + s2 + s3);
      }
      
    } // epoch
  } // train()

  // cross entropy error not applicable to regression problems

  meanSqErr(dataX, dataY)
  {
    let sumSE = 0.0;
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let y = dataY[i];  // target output like [2.3] as matrix
      let oupt = this.eval(X);  // computed like [2.07]

      for (let k = 0; k < this.no; ++k) {
        let err = y[k] - oupt[k];
      }
      let err = y[0] - oupt[0];
      sumSE += err * err;
    }
    return sumSE / dataX.length;
  } 

  accuracy(dataX, dataY, pctClose)
  {
    // correct if predicted is within pctClose of target
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
  console.log("\nBegin Boston Data regression ");

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
  let lrnRate = 0.01;
  let maxEpochs = 5000;
  console.log("\nStarting training with learning rate = 0.01 ");
  nn.train(trainX, trainY, lrnRate, maxEpochs);
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

  console.log("\nPredicted median house price of town: ");
  U.vecShow(predicted, 6, 1);  // predicted is a vector

  let predPrice = predicted[0] * 10000;
  console.log("( $" + predPrice.toFixed(2).toString() + " )" );

  process.stdout.write("\033[0m");  // reset
  console.log("\nEnd demo");
} // main()

main();


