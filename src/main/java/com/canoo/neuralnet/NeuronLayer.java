package com.canoo.neuralnet;

/**
 * Created by codecamp on 14/04/16.
 */
public class NeuronLayer {

    double[][] weights;

    // public List<List<Double>> synapticWeights = new ArrayList<>();

    public NeuronLayer(int numberOfNeurons, int numberOfInputsPerNeuron) {
        weights = new double[numberOfInputsPerNeuron][numberOfNeurons];

        for (int i = 0; i < numberOfInputsPerNeuron; ++i) {
            for (int j = 0; j < numberOfNeurons; ++j) {
                weights[i][j] = (2 * Math.random()) - 1; // shift the range from 0-1 to -1 to 1
            }
        }
    }

    public void adjustWeights(double[][] adjustment) {
        this.weights = NNMath.matrixAdd(weights, adjustment);
    }

    @Override
    public String toString() {
        return MatrixUtil.matrixToString(weights);
    }
}
