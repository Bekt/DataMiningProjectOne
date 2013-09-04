package ml.projectone;

import ml.MLException;
import ml.Matrix;
import ml.MatrixReloaded;
import ml.SupervisedLearner;

import java.util.ArrayList;
import java.util.List;

public class BaselineLearner extends SupervisedLearner {

    private List<Double> columnMeans = new ArrayList<Double>();
    private MatrixReloaded features, labels;

    @Override
    public void train(Matrix featuress, Matrix labelss) {

        // TODO: fix this shit
        MatrixReloaded features = new MatrixReloaded();
        MatrixReloaded labels = new MatrixReloaded();

        this.features = features;
        this.labels = labels;

        for (int i = 0; i < labels.getNumCols(); i++) {
            if (labels.isCategorical(i)) {
                throw new MLException("Cannot calculate mean for categorical column: "
                        + labels.getColumnAttributes(i).getName());
            }
            columnMeans.add(labels.columnMean(i));
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
     * @return Vector based SSE
     */
    @Override
    public double getAccuracy() {
        double sum = 0;
        for (int i = 0; i < labels.getNumRows(); i++) {
            List<Double> result = predict(features.getRow(i));

            if (labels.getNumCols() != result.size()) {
                throw new MLException("Returned result size is different than number of columns in labels!");
            }

            double magnitude = 0;
            for (int j = 0; j < labels.getNumCols(); j++) {
                double res = labels.getRow(i).get(j) - result.get(j);
                magnitude += res * res;

            }
            sum += magnitude;
        }
        return sum;
    }

}
