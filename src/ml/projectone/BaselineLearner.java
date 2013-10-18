package ml.projectone;

import ml.MLException;
import ml.Matrix;
import ml.SupervisedLearner;

import java.util.ArrayList;
import java.util.List;

public class BaselineLearner extends SupervisedLearner {

    private List<Double> columnMeans = new ArrayList<Double>();
    private Matrix features, labels;

    @Override
    public void train(Matrix features, Matrix labels) {

        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features and labels must have the same number of rows.");
        }

        this.features = features;
        this.labels = labels;

        for (int i = 0; i < labels.getNumCols(); i++) {
            if (labels.isCategorical(i)) {
                columnMeans.add(labels.mostCommonValue(i));
            } else {
                columnMeans.add(labels.columnMean(i));
            }
        }
    }

    @Override
    public List<Double> predict(List<Double> in) {

        if (features == null || labels == null) {
            throw new MLException("Baseline has not been trained yet!");
        }
        if (in.size() != features.getNumCols()) {
            throw new MLException("Number of columns is different than the trained data!");
        }

        List<Double> out = new ArrayList<Double>();
        for (double mean : columnMeans) {
            out.add(mean);
        }

        return out;
    }

    /**
     * Performs a n-fold cross validation and returns MSE of all samples
     * @param features
     * @param labels
     * @param n
     * @return MSE of all samples
     */
    public double nFoldCrossValidation(Matrix features, Matrix labels, int n) {

        int rows = features.getNumRows();

        if (n > rows) {
            throw new MLException(String.format(
                    "n [%d] must be <= row size [%d]", n, features.getNumRows()));
        }

        if (rows != labels.getNumRows()) {
            throw new MLException("Features and labels rows mismatch");
        }

        int foldSize = rows % n == 0 ? rows / n : rows / n + 1;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            int foldStart = i * foldSize;
            int foldEnd = (i + 1) * foldSize;
            if (foldEnd > rows) {
                foldEnd = rows;
            }

            Matrix toTrainFeatures = new Matrix(features);
            Matrix toTrainLabels = new Matrix(labels);

            Matrix toPredictFeatures = toTrainFeatures.removeFold(foldStart, foldEnd);
            Matrix toPredictLabels = toTrainLabels.removeFold(foldStart, foldEnd);

            BaselineLearner learner = new BaselineLearner();
            learner.train(toTrainFeatures, toTrainLabels);
            sum += learner.getAccuracy(toPredictFeatures, toPredictLabels);
        }
        return sum / rows;
    }

}
