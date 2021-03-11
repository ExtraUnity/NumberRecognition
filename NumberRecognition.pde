/*
Link til github repository: https://github.com/ExtraUnity/NumberRecognition
 */

import java.io.*;
import java.util.Random;

DataSet trainingSet;
DataSet testingSet;
DataSet wrongAnswers;
NeuralNetwork network;

static ArrayList<Double> meanSquaredErrors;

void setup() {

  size(850, 620);
  background(255);
  frameRate(24);
  fill(0);
  meanSquaredErrors = new ArrayList<Double>();
  try {

    network = new NeuralNetwork(784, 300, 100, 10);
    trainingSet = createTrainingSet(0, 10000);
    testingSet = createTestSet(0, 10000);
    trainData(20, 50, 10000/50, 5);
    String path = dataPath("");
    println(path);
    
    /*
    Use this line to load a network from the network.txt file
    network = NeuralNetwork.loadNetwork(path + "\\saves\\network.txt");
     */
    testData();
  } 
  catch(Exception e) {
    println(e);
  }
}

void draw() {
  PImage image = createImage(28, 28, ALPHA);

  for (int i = 0; i<image.pixels.length; i++) {
    image.pixels[i] = color((int)(wrongAnswers.getInput(frameCount-1)[i]*256));
  }

  network.feedForward(wrongAnswers.getInput(frameCount-1), 0);
  int guess = getIndexOfLargest(network.activation[network.NETWORK_SIZE-1]);

  image.resize(50, 50);
  int x = ((frameCount-1)%17)*50;
  int y = ((frameCount-1)/17)*62;
  image(image, x, y);
  text("g="+guess+" l="+(int)wrongAnswers.getOutput(frameCount-1)[0], x, y+59);
  if (frameCount==wrongAnswers.data.size()) {
    noLoop();
  }
}

double[] createLabels(int i, int size) {
  double[] tempLabels = new double[size];
  for (int j = 0; j<size; j++) {
    tempLabels[j] = i==j ? 1 : 0;
  }
  return tempLabels;
}


DataSet createTrainingSet(int lower, int upper) {
  DataSet set = new DataSet(784, 10); //input size output size
  int[][] trainingImages;
  int[] trainingLabels;
  try {
    String path = dataPath("");
    trainingImages = MnistReader.loadMnistImages(new File(path + "\\train-images.idx3-ubyte")); 
    trainingLabels = MnistReader.loadMnistLabels(new File(path +"\\train-labels.idx3-ubyte")); 
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
  int[][] testingImages;
  int[] testingLabels;
  try {
    String path = dataPath("");
    testingImages = MnistReader.loadMnistImages(new File(path + "\\t10k-images.idx3-ubyte"));
    testingLabels = MnistReader.loadMnistLabels(new File(path + "\\t10k-labels.idx3-ubyte"));
    for (int i = lower; i<upper; i++) {
      double[] input = new double[784];
      double[] target = new double[1];

      target[0] = testingLabels[i];

      for (int j = 0; j<testingImages[i].length; j++) {
        input[j] = ((double)testingImages[i][j]) / ((double)256);
      }
      set.addData(input, target);
    }
  } 
  catch(Exception e) {
    println(e);
  }

  return set;
}

void trainData(int epochs, int loops, int batch_size, int stopThreshold) throws IOException {
  float bestTest = 0;
  int wrongTurns = 0;
  ArrayList<Float> tests = new ArrayList<Float>();
  for (int e = 0; e < epochs; e++) {
    network.train(trainingSet, loops, batch_size);
    System.out.println("Epoch:  " + (e+1) + "  Out of:  " + epochs);
    float test = testData();
    tests.add(test);
    String path = dataPath("");
    if (test<bestTest) {
      wrongTurns++;
      if (wrongTurns == 0 || wrongTurns == stopThreshold) {//Early stopping to prevent overfitting
        println("breaking");

        network.saveNetwork(path + "\\saves\\network.txt");
        saveArrayList(path + "\\saves\\meanSquaredError.txt", meanSquaredErrors);
        saveArrayList(path + "\\saves\\tests.txt", tests);
        noLoop();
        break;
      }
    } else {
      bestTest = test;
      wrongTurns = 0;
      network.saveNetwork(path + "\\saves\\network.txt");
    }
    saveArrayList(path + "\\saves\\meanSquaredError.txt", meanSquaredErrors);
    saveArrayList(path + "\\saves\\tests.txt", tests);
  }
}

float testData() {
  wrongAnswers = new DataSet(784, 1);
  int correct = 0;
  int wrong = 0;
  for (int i = 0; i<testingSet.data.size(); i++) {
    double[] result = network.feedForward(testingSet.getInput(i), 0);
    if (getIndexOfLargest(result)==testingSet.getOutput(i)[0]) {
      correct++;
    } else {
      wrong++;
      wrongAnswers.data.add(testingSet.data.get(i));
    }
  }
  println("Final test accuracy: " + ((1f*correct)/(1f*(correct+wrong)))*100 + "%");
  return ((1f*correct)/(1f*(correct+wrong)));
}

int getIndexOfLargest(double[] a) {
  int indexMax = 0;
  for (int i = 0; i<a.length; i++) {
    indexMax = a[i] > a[indexMax] ? i : indexMax;
  }
  return indexMax;
}


//taken from https://stackoverflow.com/questions/16111496/java-how-can-i-write-my-arraylist-to-a-file-and-read-load-that-file-to-the
void saveArrayList(String fileName, ArrayList list) throws FileNotFoundException {
  PrintWriter pw = new PrintWriter(new FileOutputStream(fileName));
  for (Object d : list)
    pw.println(d);
  pw.close();
}

static int[] getRandomValues(int lower, int upper, int size) {
  Random indexGenerator = new Random();
  int[] is = new int[size];
  for (int i = 0; i< size; i++) {
    int n = indexGenerator.nextInt((upper-lower)) + lower;
    while (containsValue(is, n)) {
      n = indexGenerator.nextInt((upper-lower)) + lower;
      ;
    }

    is[i] = n;
  }
  return is;
}

static boolean containsValue(int[] a, int n) {
  if (a == null) return false;
  for (int i : a) {
    if (i==n) return true;
  }
  return false;
}
