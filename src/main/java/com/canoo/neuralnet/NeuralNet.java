package com.canoo.neuralnet;

import static com.canoo.neuralnet.MatrixUtil.apply;
import static com.canoo.neuralnet.NNMath.*;

/**
 * Created by codecamp on 14/04/16.
 */
public class NeuralNet {
    private final NeuronLayer layer1, layer2;
    private double[][] outputLayer1;
    private double[][] outputLayer2;

    public NeuralNet(NeuronLayer layer1, NeuronLayer layer2) {
        this.layer1 = layer1;
        this.layer2 = layer2;
    }

    /**
     * Forward propagation
     * <p>
     * Output of neuron = 1 / (1 + e^(-(sum(weight, input)))
     *
     * @param inputs
     */
    public void think(double[][] inputs) {
        outputLayer1 = apply(matrixMultiply(inputs, layer1.weights), NNMath::sigmoid); // 4x4
        outputLayer2 = apply(matrixMultiply(outputLayer1, layer2.weights), NNMath::sigmoid); // 4x1
    }

    public void train(double[][] inputs, double[][] outputs, int numberOfTrainingIterations) {
        for (int i = 0; i < numberOfTrainingIterations; ++i) {
            // pass the training set through the network
            think(inputs); // 4x3

            // adjust weights by error * input * output * (1 - output)

            // calculate the error for layer 2
            // (the difference between the desired output and predicted output for each of the training inputs)
            double[][] errorLayer2 = matrixSubtract(outputs, outputLayer2); // 4x1
            double[][] deltaLayer2 = scalarMultiply(errorLayer2, apply(outputLayer2, NNMath::sigmoidDerivative)); // 4x1

            // calculate the error for layer 1
            // (by looking at the weights in layer 1, we can determine by how much layer 1 contributed to the error in layer 2)

            double[][] errorLayer1 = matrixMultiply(deltaLayer2, matrixTranspose(layer2.weights)); // 4x4
            double[][] deltaLayer1 = scalarMultiply(errorLayer1, apply(outputLayer1, NNMath::sigmoidDerivative)); // 4x4

            // Calculate how much to adjust the weights by

            double[][] adjustmentLayer1 = matrixMultiply(matrixTranspose(inputs), deltaLayer1); // 4x4
            double[][] adjustmentLayer2 = matrixMultiply(matrixTranspose(outputLayer1), deltaLayer2); // 4x1

            // adjust the weights
            this.layer1.adjustWeights(adjustmentLayer1);
            this.layer2.adjustWeights(adjustmentLayer2);

            // if you only had one layer
            // synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
            // double[][] errorLayer1 = NNMath.matrixSubtract(outputs, outputLayer1);
            // double[][] deltaLayer1 = NNMath.matrixMultiply(errorLayer1, MatrixUtil.apply(outputLayer1, NNMath::sigmoidDerivative));
            // double[][] adjustmentLayer1 = NNMath.matrixMultiply(NNMath.matrixTranspose(inputs), deltaLayer1);

        }
    }

    public double[][] getOutput() {
        return outputLayer2;
    }

    @Override
    public String toString() {
        String result = "Layer 1\n";
        result += layer1.toString();
        result += "Layer 2\n";
        result += layer2.toString();

        if (outputLayer1 != null) {
            result += "Layer 1 output\n";
            result += MatrixUtil.matrixToString(outputLayer1);
        }

        if (outputLayer2 != null) {
            result += "Layer 2 output\n";
            result += MatrixUtil.matrixToString(outputLayer2);
        }

        return result;
    }
}
