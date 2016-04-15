package com.canoo.neuralnet.examples;

import com.canoo.neuralnet.NNMath;
import com.canoo.neuralnet.NeuralNet;
import com.canoo.neuralnet.NeuronLayer;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

/**
 * Created by fabian on 14.04.16.
 */
public class LearnIris {

    /**
     * The goal of this neural net example is to learn the add function by examples
     * category.g.
     * <p>
     * 0.5  0.45 -> 0.95
     *
     * @param args
     */
    public static void main(String args[]) {

        // create hidden layer that has 4 neurons and 4 inputs per neuron
        NeuronLayer layer1 = new NeuronLayer(NeuronLayer.InitialWeightType.RANDOM, 4, 4);

        // create output layer that has 3 neurons representing the prediction and 4 inputs for this neuron
        // (mapped from the previous hidden layer)
        NeuronLayer layer2 = new NeuronLayer(NeuronLayer.InitialWeightType.RANDOM, 1, 4);

        NeuralNet net = new NeuralNet(layer1, layer2);

        List<Plant> plants = readFile();

        Collections.shuffle(plants);

        int trainingSetSize = (int) Math.floor(plants.size() * 0.65);

        double[][] inputs = new double[trainingSetSize][4];
        double[][] outputs = new double[trainingSetSize][3];
        for (int i = 0; i < trainingSetSize; i++) {
            Plant plant = plants.get(i);
            inputs[i] = NNMath.normalize(new double[]{plant.a, plant.b, plant.c, plant.d});
            //inputs[i] = new double[]{plant.a, plant.b, plant.c, plant.d};
            if(plant.category == 0) {
                outputs[i] = new double[]{0.0};
            } else if(plant.category == 1) {
                outputs[i] = new double[]{0.5};
            } else {
                outputs[i] = new double[]{1.0};
            }

        }


        System.out.println("Training the neural net...");
        net.train(inputs, outputs, 50);
        System.out.println("Finished training");

        System.out.println("Layer 1 weights");
        System.out.println(layer1);

        System.out.println("Layer 2 weights");
        System.out.println(layer2);

        // calculate the predictions on unknown data
        int successful = 0;
        for (int j = trainingSetSize; j < plants.size() ; j++) {
            Plant plant = plants.get(j);
            //double[][] testInput = {NNMath.normalize(new double[]{plant.a, plant.b, plant.c, plant.d})};
            double[][] testInput = {new double[]{plant.a, plant.b, plant.c, plant.d}};
            boolean success = predict(testInput, plant.category, net);
            if(success){
                successful++;
            }
        }
        System.out.println("Correctly predicted " + successful +" out of " + (plants.size() - trainingSetSize));

    }

    private static List<Plant> readFile() {
        List<Plant> plants = new ArrayList<>();

        try (Stream<String> stream = Files.lines(Paths.get(LearnIris.class.getResource("iris.txt").toURI()))) {
            stream.forEach(l -> {
                String[] parts = l.split(",");
                Plant plant = new Plant(Double.parseDouble(parts[0]), Double.parseDouble(parts[1]), Double.parseDouble(parts[2]), Double.parseDouble(parts[3]), Integer.parseInt(parts[4]));
                plants.add(plant);
            });

        } catch (IOException | URISyntaxException e) {
            e.printStackTrace();
        }
        return plants;
    }

    public static boolean predict(double[][] testInput, double expected, NeuralNet net) {
        net.think(testInput);

        // then
        System.out.println("Prediction on data "
                + format(testInput[0][0]) + " "
                + format(testInput[0][1]) + " "
                + format(testInput[0][2]) + " "
                + format(testInput[0][3]) + ": "
                + format(net.getOutput()[0][0])+ ", expected -> " +  expected/2 + " ");

        return expected/2-0.1 < net.getOutput()[0][0] && net.getOutput()[0][0] <expected/2+0.1;
    }

    private static String format(double x){
        return String.format("%.03f",x);

    }

    private static class Plant {
        double a;
        double b;
        double c;
        double d;
        int category;


        public Plant(double a, double b, double c, double d, int category) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
            this.category = category;
        }
    }
}
