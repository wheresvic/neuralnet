package com.canoo.neuralnet.examples;

import com.canoo.neuralnet.MatrixUtil;
import com.canoo.neuralnet.NeuralNet;
import com.canoo.neuralnet.NeuronLayer;

/**
 * Created by codecamp on 14/04/16.
 */
public class LearnFirstColumn {

    /**
     * The goal of this neural net example is to train it on a small dataset where the output
     * is simply the value of the first input parameter
     *
     * e.g.
     *
     * 0 0 1 -> 0
     * 1 1 1 -> 1
     * 1 0 1 -> 1
     * 0 1 1 -> 0
     *
     * @param args
     */
    public static void main(String args[]) {

        // create hidden layer that has 4 neurons and 3 inputs per neuron
        NeuronLayer layer1 = new NeuronLayer(NeuronLayer.InitialWeightType.RANDOM, 4, 3);

        // create output layer that has 1 neuron representing the prediction and 4 inputs for this neuron
        // (mapped from the previous hidden layer)
        NeuronLayer layer2 = new NeuronLayer(NeuronLayer.InitialWeightType.RANDOM, 1, 4);

        NeuralNet net = new NeuralNet(layer1, layer2);

        // train the net
        double[][] inputs = new double[][]{
                {0, 0, 1},
                {1, 1, 1},
                {1, 0, 1},
                {0, 1, 1}
        };

        double[][] outputs = new double[][]{
                {0},
                {1},
                {1},
                {0}
        };

        System.out.println("Training the neural net...");
        net.train(inputs, outputs, 10000);
        System.out.println("Finished training");

        System.out.println("Layer 1 weights");
        System.out.println(layer1);

        System.out.println("Layer 2 weights");
        System.out.println(layer2);

        // calculate the predictions on unknown data

        // 1, 0, 0
        predict(new double[][]{{1, 0, 0}}, net);

        // 0, 0, 0
        predict(new double[][]{{0, 0, 0}}, net);

        // 0, 1, 0
        predict(new double[][]{{0, 1, 0}}, net);

        // 1, 1, 0
        predict(new double[][]{{1, 1, 0}}, net);
    }

    public static void predict(double[][] testInput, NeuralNet net) {
        net.think(testInput);

        // then
        System.out.println("Prediction on data "
                + testInput[0][0] + " "
                + testInput[0][1] + " "
                + testInput[0][2] + " -> "
                + net.getOutput()[0][0] + ", expected -> " + testInput[0][0]);
    }
}
