package ml.projectone;

import ml.MLException;
import ml.Matrix;
import ml.SupervisedLearner;

import java.util.ArrayList;
import java.util.List;

public class BaselineLearner extends SupervisedLearner {

    private List<Double> columnMeans;
    private Matrix features, labels;

    @Override
    public void train(Matrix features, Matrix labels) {

        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features and labels must have the same number of rows.");
        }

        this.features = features;
        this.labels = labels;
        this.columnMeans = new ArrayList<Double>();

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

}
