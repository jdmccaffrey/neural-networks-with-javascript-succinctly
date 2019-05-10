// iris_nn.js
// ES6

let U = require("../Utilities/utilities_lib.js");

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
    //console.log("\nHidden node values: ");  
    //vecShow(this.hNodes, 4);

    for (let k = 0; k < this.no; ++k) {
      for (let j = 0; j < this.nh; ++j) {
        oSums[k] += this.hNodes[j] * this.hoWeights[j][k];
      }
      oSums[k] += this.oBiases[k];
    }

    this.oNodes = U.softmax(oSums);
    // console.log("\nPre-softmax output nodes: ");
    // vecShow(this.oNodes, 4);

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
    let numWts = (this.ni * this.nh) + this.nh + (this.nh * this.no) + this.no;
    let result = vecMake(numWts, 0.0);
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

    let n = trainX.length;  // 120
    let indices = U.arange(n);  // [0,1,..,119]
    let freq = Math.trunc(maxEpochs / 10);  // when to print error
    
    for (let epoch = 0; epoch < maxEpochs; ++epoch) {
      this.shuffle(indices);  //
      for (let ii = 0; ii < n; ++ii) {  // each item
        let idx = indices[ii];
        let X = trainX[idx];
        let Y = trainY[idx];
        this.eval(X);  // output stored in this.oNodes

        // compute output node signals
        for (let k = 0; k < this.no; ++k) {
          let derivative = (1 - this.oNodes[k]) * this.oNodes[k];  // assumes softmax
          oSignals[k] = derivative * (this.oNodes[k] - Y[k]);  // E=(t-o)^2 do E'=(o-t) 
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

  meanCrossEntErr(dataX, dataY)
  {
    let sumCEE = 0.0;  // sum of the cross entropy errors
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let Y = dataY[i];  // target output like (0, 1, 0)
      let oupt = this.eval(X);  // computed like (0.23, 0.66, 0.11)
      let idx = U.argmax(Y);  // find location of the 1 in target
      sumCEE += Math.log(oupt[idx]);
    }
    sumCEE *= -1;
    return sumCEE / dataX.length;
  }

  meanSqErr(dataX, dataY)
  {
    let sumSE = 0.0;
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let Y = dataY[i];  // target output like (0, 1, 0)
      let oupt = this.eval(X);  // computed like (0.23, 0.66, 0.11)
      for (let k = 0; k < this.no; ++k) {
        let err = Y[k] - oupt[k]  // target - computed
        sumSE += err * err;
      }
    }
    return sumSE / dataX.length;
  } // meanSqErr()

  accuracy(dataX, dataY)
  {
    let nc = 0; let nw = 0;
    for (let i = 0; i < dataX.length; ++i) {  // each data item
      let X = dataX[i];
      let Y = dataY[i];  // target output like (0, 1, 0)
      let oupt = this.eval(X);  // computed like (0.23, 0.66, 0.11)
      let computedIdx = U.argmax(oupt);
      let targetIdx = U.argmax(Y);
      if (computedIdx == targetIdx) {
        ++nc;
      }
      else {
        ++nw;
      }
    }
    return nc / (nc + nw);
  } // accuracy()


} // NeuralNet

// =============================================================================

function main()
{
  process.stdout.write("\033[0m");  // reset
  process.stdout.write("\x1b[1m" + "\x1b[37m");  // bright white
  console.log("\nBegin NN back-prop with JavaScript demo ");

  // 1. load data
  // data looks like: 5.1, 3.5, 1.4, 0.2, 1, 0, 0
  let trainX = U.loadTxt(".\\Data\\iris_train.txt", ",", [0,1,2,3]);
  let trainY = U.loadTxt(".\\Data\\iris_train.txt", ",", [4,5,6]);
  let testX = U.loadTxt(".\\Data\\iris_test.txt", ",", [0,1,2,3]);
  let testY = U.loadTxt(".\\Data\\iris_test.txt", ",", [4,5,6]);

  // 2. create network
  console.log("\nCreating a 4-7-3 tanh, softmax NN for Iris dataset");
  let seed = 0;
  let nn = new NeuralNet(4, 7, 3, seed);

  // 3. train network
  let lrnRate = 0.01;
  let maxEpochs = 50;
  console.log("\nStarting training with learning rate = 0.01 ");
  nn.train(trainX, trainY, lrnRate, maxEpochs);
  console.log("Training complete");

  // 4. evaluate model
  let trainAcc = nn.accuracy(trainX, trainY);
  let testAcc = nn.accuracy(testX, testY);
  console.log("\nAccuracy on training data = " +
    trainAcc.toFixed(4).toString()); 
  console.log("Accuracy on test data     = " +
    testAcc.toFixed(4).toString());

  // 5. use trained model
  let unknown = [5.1, 3.1, 4.1, 2.1];  // set., set., ver., vir.
  let predicted = nn.eval(unknown);
  console.log("\nSetting features of iris to predict: ");
  U.vecShow(unknown, 1, 12);
  console.log("\nPredicted quasi-probabilities: ");
  U.vecShow(predicted, 4, 12); 

  process.stdout.write("\033[0m");  // reset
  console.log("\nEnd demo");
} // main()

main();
