// utilities_lib.js
// ES6

let FS = require('fs');

function loadTxt(fn, delimit, usecols) {
  let all = FS.readFileSync(fn, "utf8");  // giant string
  all = all.trim();  // strip final crlf in file
  let lines = all.split("\n");
  let rows = lines.length;
  let cols = usecols.length;
  let result = matMake(rows, cols, 0.0); 
  for (let i = 0; i < rows; ++i) {  // each line
    let tokens = lines[i].split(delimit);
    for (let j = 0; j < cols; ++j) {
      result[i][j] = parseFloat(tokens[usecols[j]]);
    }
  }
  return result;
}

function arange(n)
{
  let result = [];
  for (let i = 0; i < n; ++i) {
    result[i] = Math.trunc(i);
  } 
  return result;
}

class Erratic
{
  constructor(seed) {
    this.seed = seed + 0.5;  // avoid 0
  }

  next() {
    let x = Math.sin(this.seed) * 1000;
    let result = x - Math.floor(x);  // [0.0,1.0)
    this.seed = result;  // for next call
    return result;
  }

  nextInt(lo, hi) {
    let x = this.next();
    return Math.trunc((hi - lo) * x + lo);
  }
}

function vecMake(n, val)
{
  let result = [];
  for (let i = 0; i < n; ++i) {
    result[i] = val;
  }
  return result;
}

function matMake(rows, cols, val)
{
  let result = [];
  for (let i = 0; i < rows; ++i) {
     result[i] = [];
     for (let j = 0; j < cols; ++j) {
       result[i][j] = val;
     }
  }
  return result;
}

function vecShow(v, dec, len)
{
  for (let i = 0; i < v.length; ++i) {
    if (i != 0 && i % len == 0) {
      process.stdout.write("\n");
    }
    if (v[i] >= 0.0) {
      process.stdout.write(" ");  // + or - space
    }
    process.stdout.write(v[i].toFixed(dec));
    process.stdout.write("  ");
  }
  process.stdout.write("\n");
}

function matShow(m, dec)
{
  let rows = m.length;
  let cols = m[0].length;
  for (let i = 0; i < rows; ++i) {
    for (let j = 0; j < cols; ++j) {
      if (m[i][j] >= 0.0) {
        process.stdout.write(" ");  // + or - space
      }
      process.stdout.write(m[i][j].toFixed(dec));
      process.stdout.write("  ");
    }
    process.stdout.write("\n");
  }
}

function argmax(vec)
{
  let result = 0;
  let m = vec[0];
  for (let i = 0; i < vec.length; ++i) {
    if (vec[i] > m) {
      m = vec[i];
      result = i;
    }
  }
  return result;
}

function hyperTan(x)
{
  if (x < -20.0) {
    return -1.0;
  }
  else if (x > 20.0) {
    return 1.0;
  }
  else {
    return Math.tanh(x);
  }
}

function logSig(x)
{
  if (x < -20.0) {
    return 0.0;
  }
  else if (x > 20.0) {
    return 1.0;
  }
  else {
    return 1.0 / (1.0 + Math.exp(-x));
  }
}

function vecMax(vec)
{
  let mx = vec[0];
  for (let i = 0; i < vec.length; ++i) {
    if (vec[i] > mx) {
      mx = vec[i];
    }
  }
  return mx;
}

function softmax(vec)
{
  //let m = Math.max(...vec);  // or 'spread' operator
  let m = vecMax(vec);
  let result = [];
  let sum = 0.0;
  for (let i = 0; i < vec.length; ++i) {
    result[i] = Math.exp(vec[i] - m);
    sum += result[i];
  }
  for (let i = 0; i < result.length; ++i) {
    result[i] = result[i] / sum;
  }
  return result;
}

module.exports = {
  vecMake,
  matMake,
  vecShow,
  matShow,
  argmax,
  loadTxt,
  arange,
  Erratic,
  hyperTan,
  logSig,
  vecMax,
  softmax,
};
