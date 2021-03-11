static class NeuralNetwork implements Serializable {
  final int[] LAYER_SIZES;
  final int INPUT_SIZE;
  final int OUTPUT_SIZE;
  final int NETWORK_SIZE;

  double[][] activation;
  double[][] bias;
  double[][][] weights; //one for layer, one for neuron, one for previous neuron
  double[][] error;


  NeuralNetwork(int... LAYER_SIZES) {
    this.LAYER_SIZES = LAYER_SIZES;
    this.INPUT_SIZE = LAYER_SIZES[0];
    this.NETWORK_SIZE = LAYER_SIZES.length;
    this.OUTPUT_SIZE = LAYER_SIZES[NETWORK_SIZE-1];

    this.activation = new double[NETWORK_SIZE][];
    this.bias = new double[NETWORK_SIZE][];
    this.weights = new double[NETWORK_SIZE][][];

    this.error = new double[NETWORK_SIZE][];
    for (int i = 0; i<NETWORK_SIZE; i++) {
      this.activation[i] = new double[LAYER_SIZES[i]];
      this.bias[i] = createRandomArray(LAYER_SIZES[i], -0.4, 0.7); //arbitrary bounds
      this.error[i] = new double[LAYER_SIZES[i]];

      if (i>0) {
        this.weights[i] = createRandomArray(LAYER_SIZES[i], LAYER_SIZES[i-1], -1, 1); //arbitrary bounds
      }
    }
  }

  double[] feedForward(double[] input, float dropoutRatio) {
    if (input.length != this.INPUT_SIZE) {
      return null;
    }
    this.activation[0] = input;

    for (int l = 1; l<NETWORK_SIZE; l++) { //all layers, l
      int[] dropouts = null;
      if (l!=NETWORK_SIZE-1) {
        dropouts = getRandomValues(0, LAYER_SIZES[l]-1, (int)(LAYER_SIZES[l]*dropoutRatio));  //25% dropout, to discourage memorization of training data
      }
      for (int j = 0; j<LAYER_SIZES[l]; j++) {//all neurons in layer
        if (!containsValue(dropouts, j)) {
          double sum = bias[l][j]; //initial value is just the bias

          for (int k = 0; k<LAYER_SIZES[l-1]; k++) {//all neurons in previous layer
            sum += activation[l-1][k] * weights[l][j][k]; //activation of previous neuron times weights from previous to current neuron
          }

          this.activation[l][j] = sigmoid(sum);
        } else {
          this.activation[l][j] = 0;
        }
      }
    }
    return activation[NETWORK_SIZE-1];
  }

  void train(DataSet set, int loops, int batchSize) {
    for (int i = 0; i<loops; i++) {
      DataSet batch = set.getBatch(batchSize);
      for (int b = 0; b<batchSize-1; b++) {
        this.train(batch.getInput(b), batch.getOutput(b), 0.3);
      }

      println(meanSquaredError(batch));
      meanSquaredErrors.add(meanSquaredError(batch));
    }
  }

  void train(double[] input, double[] target, double learningRate) {
    if (input.length == INPUT_SIZE && target.length == OUTPUT_SIZE) {
      feedForward(input, 0.25);
      backpropError(target);
      update(learningRate);
    }
  }

  double meanSquaredError(double[] input, double[] target) {
    feedForward(input, 0.25);
    double e = 0;
    for (int i = 0; i<target.length; i++) {
      e+=(target[i]-activation[NETWORK_SIZE-1][i])*(target[i]-activation[NETWORK_SIZE-1][i]);
    }
    e /= 2d*target.length; //we use half of the mean squared error
    return e;
  }

  double meanSquaredError(DataSet set) {
    double e = 0;
    for (int i = 0; i<set.data.size(); i++) {
      e += meanSquaredError(set.getInput(i), set.getOutput(i));
    }
    e/=set.data.size();
    return e;
  }

  void backpropError(double[] target) {
    for (int i = 0; i<LAYER_SIZES[NETWORK_SIZE-1]; i++) {

      error[NETWORK_SIZE-1][i] = (activation[NETWORK_SIZE-1][i]-target[i]) * derivSigmoid(activation[NETWORK_SIZE-1][i]);
    }

    for (int l = NETWORK_SIZE-2; l>0; l--) {
      for (int j = 0; j<LAYER_SIZES[l]; j++) {
        double sum = 0;
        for (int k = 0; k<LAYER_SIZES[l+1]; k++) {
          sum += weights[l+1][k][j] * error[l+1][k];
        }
        error[l][j] = sum*activation[l][j]*(1-activation[l][j]);
      }
    }
  }

  void update(double learningRate) {
    for (int l = 1; l<NETWORK_SIZE; l++) {
      for (int j = 0; j<LAYER_SIZES[l]; j++) {
        for (int k = 0; k<LAYER_SIZES[l-1]; k++) {
          weights[l][j][k] -= learningRate*activation[l-1][k]*error[l][j];
        }
        bias[l][j] -= learningRate*error[l][j]; //as bias is not connected to previous, we just set the activtion to 1 which cancels out
      }
    }
  }

  double sigmoid(double num) {
    return 1d/(1+Math.exp(-num));
  }

  double derivSigmoid(double num) {
    return num*(1-num);
  }

  double[] createRandomArray(int size, double lower, double upper) {
    double[] array = new double[size];
    for (int i = 0; i<size; i++) {
      array[i] = Math.random()*(upper-lower)+lower;
    }
    return array;
  }

  double[][] createRandomArray(int sizeX, int sizeY, double lower, double upper) {
    double[][] array = new double[sizeX][sizeY];
    for (int i = 0; i< sizeX; i++) {
      array[i] = createRandomArray(sizeY, lower, upper);
    }
    return array;
  }

   void saveNetwork(String file) throws IOException {
    File f = new File(file);
    ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(f));
    out.writeObject(this);
    out.flush();
    out.close();
    println("network saved");
  }

  static NeuralNetwork loadNetwork(String file) throws IOException, ClassNotFoundException {

    File f = new File(file);
    ObjectInputStream in = new ObjectInputStream(new FileInputStream(f));
    NeuralNetwork network = (NeuralNetwork) in.readObject();
    in.close();
    return network;
  }
}
