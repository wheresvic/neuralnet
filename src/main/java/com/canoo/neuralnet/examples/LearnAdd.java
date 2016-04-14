package com.canoo.neuralnet.examples;

import com.canoo.neuralnet.NeuralNet;
import com.canoo.neuralnet.NeuronLayer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class LearnAdd {

    /**
     * The goal of this neural net example is to learn the add function by examples
     * e.g.
     * <p>
     * 0.5  0.45 -> 0.95
     *
     * @param args
     */
    public static void main(String args[]) {

        // create hidden layer that has 4 neurons and 2 inputs per neuron
        NeuronLayer layer1 = new NeuronLayer(NeuronLayer.InitialWeightType.RANDOM, 4, 2);

        // create output layer that has 1 neuron representing the prediction and 4 inputs for this neuron
        // (mapped from the previous hidden layer)
        NeuronLayer layer2 = new NeuronLayer(NeuronLayer.InitialWeightType.RANDOM, 1, 4);

        NeuralNet net = new NeuralNet(layer1, layer2);

        List<Tuple> tuples = createTrainingSet(20, 5);

        double[][] inputs = new double[tuples.size()][2];
        double[][] outputs = new double[tuples.size()][1];
        for (int i = 0; i < tuples.size(); i++) {
            inputs[i][0] = tuples.get(i).a;
            inputs[i][1] = tuples.get(i).b;
            outputs[i][0] = inputs[i][0] + inputs[i][1];
        }


        System.out.println("Training the neural net...");
        net.train(inputs, outputs, 10000);
        System.out.println("Finished training");

        System.out.println("Layer 1 weights");
        System.out.println(layer1);

        System.out.println("Layer 2 weights");
        System.out.println(layer2);

        // calculate the predictions on unknown data


        predict(new double[][]{{0.25, 0.1}}, net);

        predict(new double[][]{{0.99, -0.33}}, net);

        predict(new double[][]{{0.2, 0.2}}, net);
    }

    private static List<Tuple> createTrainingSet(int trainingSetSize, int seed) {

        Random random = new Random(seed);
        List<Tuple> tuples = new ArrayList<>();

        for (int i = 0; i < trainingSetSize; i++) {
            double s1 = random.nextDouble()*0.5;
            double s2 = random.nextDouble()*0.5;
            tuples.add(new Tuple(s1, s2));
        }
        return tuples;
    }

    public static void predict(double[][] testInput, NeuralNet net) {
        net.think(testInput);

        // then
        System.out.println("Prediction on data "
                + testInput[0][0] + " "
                + testInput[0][1] + " "
                + net.getOutput()[0][0] + ", expected -> " +   (testInput[0][0] + testInput[0][1])+ " ");
    }

    private static class Tuple {
        double a;
        double b;


        Tuple(double b, double a) {
            this.b = b;
            this.a = a;
        }
    }
}
