package com.canoo.neuralnet;

/**
 * Created by codecamp on 14/04/16.
 */
public class NeuronLayer {

    public enum InitialWeightType {
        RANDOM // only support random for the moment
    }

    double[][] weights;

    // public List<List<Double>> synapticWeights = new ArrayList<>();

    public NeuronLayer(int numberOfNeurons, int numberOfInputsPerNeuron) {
        this(InitialWeightType.RANDOM, numberOfNeurons, numberOfInputsPerNeuron);
    }

    public NeuronLayer(InitialWeightType initialWeightType, int numberOfNeurons, int numberOfInputsPerNeuron) {
        weights = new double[numberOfInputsPerNeuron][numberOfNeurons];

        for (int i = 0; i < numberOfInputsPerNeuron; ++i) {
            for (int j = 0; j < numberOfNeurons; ++j) {
                if (InitialWeightType.RANDOM == initialWeightType) {
                    weights[i][j] = (2 * Math.random()) - 1; // shift the range from 0-1 to -1 to 1
                }
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
