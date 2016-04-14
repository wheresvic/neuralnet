package com.canoo.neuralnet;

import com.canoo.neuralnet.MatrixUtil;
import com.canoo.neuralnet.NeuralNet;
import com.canoo.neuralnet.NeuronLayer;
import org.junit.Test;

/**
 * Created by codecamp on 14/04/16.
 */
public class NeuralNetTest {

    @Test
    public void shouldPrintNeuralNet() {
        // when
        NeuronLayer layer1 = new NeuronLayer(4, 3);
        NeuronLayer layer2 = new NeuronLayer(1, 4);
        NeuralNet net = new NeuralNet(layer1, layer2);

        // then
        System.out.println(net);
    }

    @Test
    public void shouldPrintNeuralNetAfterThinking() {
        // when
        NeuronLayer layer1 = new NeuronLayer(4, 3);
        NeuronLayer layer2 = new NeuronLayer(1, 4);
        NeuralNet net = new NeuralNet(layer1, layer2);

        double[][] inputs = new double[][]{
                {0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}
        };

        net.think(inputs);

        // then
        System.out.println(net);
    }

    @Test
    public void shouldPrintNeuralNetAfterTraining() {
        // when
        NeuronLayer layer1 = new NeuronLayer(4, 3);
        NeuronLayer layer2 = new NeuronLayer(1, 4);
        NeuralNet net = new NeuralNet(layer1, layer2);

        double[][] inputs = new double[][]{
                {0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}
        };

        double[][] outputs = new double[][]{
                {0},
                {1},
                {1},
                {0}
        };

        net.train(inputs, outputs, 1);

        // then
        System.out.println(net);
    }

    @Test
    public void shouldPrintNeuralNetAfterTrainingMultipleIterations() {
        // when
        NeuronLayer layer1 = new NeuronLayer(4, 3);
        NeuronLayer layer2 = new NeuronLayer(1, 4);
        NeuralNet net = new NeuralNet(layer1, layer2);

        double[][] inputs = new double[][]{
                {0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}
        };

        double[][] outputs = new double[][]{
                {0},
                {1},
                {1},
                {0}
        };

        net.train(inputs, outputs, 10000);

        double[][] testInput = new double[][]{
                {1, 0, 0}
        };

        net.think(testInput);

        // then
        System.out.println(MatrixUtil.matrixToString(net.getOutput()));
    }
}