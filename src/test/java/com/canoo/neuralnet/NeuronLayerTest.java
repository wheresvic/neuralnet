package com.canoo.neuralnet;

import com.canoo.neuralnet.NeuronLayer;
import org.junit.Test;

/**
 * Created by codecamp on 14/04/16.
 */
public class NeuronLayerTest {

    @Test
    public void shouldInitializeWeights() {
        // when
        NeuronLayer sut = new NeuronLayer(3, 4);

        // then
        System.out.println(sut);
    }

}