import java.io.*;

PImage image;
int[][] trainingImages;
int[] trainingLabels;
double[][] input;
double[][] labels;
DataSet trainingSet;
DataSet testingSet;
DataSet wrongAnswers;
NeuralNetwork network;
void setup() {
  size(200, 200);
  frameRate(0.5);
  
  try {
    network = new NeuralNetwork(784, 300, 100, 10);
    trainingSet = createTrainingSet(0, 60000);
    //trainData(50, 50, 1200);
    String path = dataPath("");
    println(path);
    network = NeuralNetwork.loadNetwork(path + "\\saves\\network.txt");
    testingSet = createTestSet(0, 10000);
    testData();
  } 
  catch(Exception e) {
    println(e);
  }
}

void draw() {
  image = createImage(28, 28, ALPHA);

  for (int i = 0; i<image.pixels.length; i++) {
    image.pixels[i] = color((int)(wrongAnswers.data.get(frameCount-1)[0][i]*256));
  }
  
  network.feedForward(wrongAnswers.data.get(frameCount-1)[0]);
  int guess = getIndexOfLargest(network.activation[network.NETWORK_SIZE-1]);
  println(guess, wrongAnswers.data.get(frameCount-1)[1][0]);

  image.resize(200, 200);
  image(image, 0, 0);
}

double[] createLabels(int i, int size) {
  double[] tempLabels = new double[size];
  for (int j = 0; j<size; j++) {

    if (j==i) {

      tempLabels[j] = 1;
    } else {
      tempLabels[j] = 0;
    }
  }
  return tempLabels;
}


DataSet createTrainingSet(int lower, int upper) {
  DataSet set = new DataSet(784, 10); //input size output size

  try {
    MnistReader fileReader = new MnistReader();
    String path = dataPath("");
    trainingImages = fileReader.loadMnistImages(new File(path + "\\train-images.idx3-ubyte")); 
    trainingLabels = fileReader.loadMnistLabels(new File(path +"\\train-labels.idx3-ubyte")); 
    for (int i = lower; i<upper; i++) {
      double[] input = new double[784];
      double[] output = new double[10];

      output = createLabels(trainingLabels[i], output.length);

      for (int j = 0; j<trainingImages[i].length; j++) {
        input[j] = ((double)trainingImages[i][j]) / ((double)256);
      }
      set.addData(input, output);
    }
  } 
  catch(Exception e) {
    println(e);
  }

  return set;
}

DataSet createTestSet(int lower, int upper) {
  DataSet set = new DataSet(784, 1); //input size output size

  try {
    MnistReader fileReader = new MnistReader();
    String path = dataPath("");
    trainingImages = fileReader.loadMnistImages(new File(path + "\\t10k-images.idx3-ubyte"));
    trainingLabels = fileReader.loadMnistLabels(new File(path + "\\t10k-labels.idx3-ubyte"));
    for (int i = lower; i<upper; i++) {
      double[] input = new double[784];
      double[] target = new double[1];

      target[0] = trainingLabels[i];

      for (int j = 0; j<trainingImages[i].length; j++) {
        input[j] = ((double)trainingImages[i][j]) / ((double)256);
      }
      set.addData(input, target);
    }
  } 
  catch(Exception e) {
    println(e);
  }

  return set;
}

void trainData(int epochs, int loops, int batch_size) throws IOException {
  for (int e = 0; e < epochs; e++) {
    network.train(trainingSet, loops, batch_size);
    System.out.println("Epoch:  " + (e+1) + "  Out of:  " + epochs);
    String path = dataPath("");
    network.saveNetwork(path + "\\saves\\network.txt");
  }
}

void testData() {
  wrongAnswers = new DataSet(784,1);
  int correct = 0;
  int wrong = 0;
  for (int i = 0; i<testingSet.data.size(); i++) {
    network.feedForward(testingSet.data.get(i)[0]);
    if (getIndexOfLargest(network.activation[network.NETWORK_SIZE-1])==testingSet.data.get(i)[1][0]) {
      correct++;
    } else {
      wrong++;
      wrongAnswers.data.add(testingSet.data.get(i));
      
    }
    println((1f*correct)/(1f*(correct+wrong)));
  }
  println("Final test accuracy: " + ((1f*correct)/(1f*(correct+wrong)))*100 + "%");
}

int getIndexOfLargest(double[] a) {
  int indexMax = 0;
  for (int i = 0; i<a.length; i++) {
    indexMax = a[i] > a[indexMax] ? i : indexMax;
  }
  return indexMax;
}
