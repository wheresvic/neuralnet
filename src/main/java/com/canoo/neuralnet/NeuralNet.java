package com.canoo.neuralnet;

import java.util.function.Function;

import static com.canoo.neuralnet.MatrixUtil.apply;
import static com.canoo.neuralnet.NNMath.*;

/**
 * https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a#.9kcfharq6
 * http://stevenmiller888.github.io/mind-how-to-build-a-neural-network-part-2/
 */
public class NeuralNet {

    public enum ActivationFunction {
        SIGMOID,
        TANH
    }

    private final NeuronLayer layer1, layer2;
    private double[][] outputLayer1;
    private double[][] outputLayer2;

    private Function<Double,Double> activationFunction;
    private Function<Double,Double> activationFunctionDerivative;


    public NeuralNet(NeuronLayer layer1, NeuronLayer layer2) {
        this.layer1 = layer1;
        this.layer2 = layer2;
        this.activationFunction = NNMath::sigmoid;
        this.activationFunctionDerivative = NNMath::sigmoidDerivative;
    }

    /**
     * Forward propagation
     * <p>
     * Output of neuron = 1 / (1 + e^(-(sum(weight, input)))
     *
     * @param inputs
     */
    public void think(double[][] inputs) {
        outputLayer1 = apply(matrixMultiply(inputs, layer1.weights), NNMath::tanh); // 4x4
        outputLayer2 = apply(matrixMultiply(outputLayer1, layer2.weights), NNMath::tanh); // 4x1
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
            // Since we’re dealing with matrices, we handle the division by multiplying the delta output sum with the inputs' transpose!

            double[][] adjustmentLayer1 = matrixMultiply(matrixTranspose(inputs), deltaLayer1); // 4x4
            double[][] adjustmentLayer2 = matrixMultiply(matrixTranspose(outputLayer1), deltaLayer2); // 4x1

            adjustmentLayer1 = MatrixUtil.apply(adjustmentLayer1, (e) -> 0.1 * e);
            adjustmentLayer2 = MatrixUtil.apply(adjustmentLayer2, (e) -> 0.1 * e);

            // adjust the weights
            this.layer1.adjustWeights(adjustmentLayer1);
            this.layer2.adjustWeights(adjustmentLayer2);

            // if you only had one layer
            // synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
            // double[][] errorLayer1 = NNMath.matrixSubtract(outputs, outputLayer1);
            // double[][] deltaLayer1 = NNMath.matrixMultiply(errorLayer1, MatrixUtil.apply(outputLayer1, NNMath::sigmoidDerivative));
            // double[][] adjustmentLayer1 = NNMath.matrixMultiply(NNMath.matrixTranspose(inputs), deltaLayer1);

            if(i % 10000 == 0){
                System.out.println(" Training iteration " + i + " of " + numberOfTrainingIterations);
            }
            //System.out.println(this);

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
