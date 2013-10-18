package ml.projectone;

import ml.ARFFParser;
import ml.Matrix;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws IOException {

        if (args.length != 1) {
            throw new IllegalArgumentException("Too few or too many arguments.");
        }

        final int featuresStart = 0;
        final int featuresEnd = 4;
        final int labelsStart = 4;
        final int labelsEnd = 5;
        final int nFoldSize = 3;
        final int repetitions = 1000;

        String filepath = args[0];

        Matrix matrix = ARFFParser.loadARFF(filepath);
        Matrix features = matrix.subMatrixCols(featuresStart, featuresEnd);
        Matrix labels = matrix.subMatrixCols(labelsStart, labelsEnd);

        Random seed = new Random();
        double mseTotal = 0;
        for (int i = 0; i < repetitions; i++) {
            BaselineLearner learner = new BaselineLearner();
            features.shuffle(seed);
            labels.shuffle(seed);
            mseTotal += learner.nFoldCrossValidation(features, labels, nFoldSize);
        }
        System.out.println("MSE: " + mseTotal / repetitions);
    }

}
