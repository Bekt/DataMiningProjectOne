package ml.projectone;

import ml.ARFFParser;
import ml.Matrix;

import java.io.FileNotFoundException;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {

        //if (args.length != 1) {
        //    throw new IllegalArgumentException("Too few or too many arguments.");
        //}

        final int featuresStart = 0;
        final int featuresEnd = 4;
        final int labelsStart = 4;
        final int labelsEnd = 5;
        final int nFoldSize = 3;
        final int repetitions = 1000;

        String filepath = "/Users/dev/workspace/DataMining2013F/iris.arff";

        Matrix matrix = ARFFParser.loadARFF(filepath);
        Matrix features = matrix.subMatrixCols(featuresStart, featuresEnd);
        Matrix labels = matrix.subMatrixCols(labelsStart, labelsEnd);

        Random seed = new Random();
        double mseTotal = 0;
        for (int i = 0; i < repetitions; i++) {
            features.shuffle(seed);
            labels.shuffle(seed);
            mseTotal += BaselineLearner.nFoldCrossValidation(features, labels, nFoldSize);
        }
        System.out.println("MSE: " + mseTotal / repetitions);
    }

}
