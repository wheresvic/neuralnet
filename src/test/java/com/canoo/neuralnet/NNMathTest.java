package com.canoo.neuralnet;

import com.canoo.neuralnet.MatrixUtil;
import com.canoo.neuralnet.NNMath;
import org.junit.Test;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

/**
 * Created by codecamp on 14/04/16.
 */
public class NNMathTest {

    @Test
    public void shouldMultiply4x2into2x3() {
        // given
        double a[][] = new double[][]{
                {2, 3},
                {4, 5},
                {6, 7},
                {8, 9}};

        double b[][] = new double[][]{
                {3, 1, 6},
                {1, 2, 4}};

        // when
        double result[][] = NNMath.matrixMultiply(a, b);

        // then
        String expected = "[[9.0 8.0 24.0 ]\n" +
                "[17.0 14.0 44.0 ]\n" +
                "[25.0 20.0 64.0 ]\n" +
                "[33.0 26.0 84.0 ]\n" +
                "]\n";

        System.out.println(MatrixUtil.matrixToString(result));
        assertThat(MatrixUtil.matrixToString(result), is(expected));
    }

    @Test
    public void shouldMultiply4x3into3x1() {
        // given
        double a[][] = new double[][]{
                {2, 3},
                {4, 5},
                {6, 7},
                {8, 9}};

        double b[][] = new double[][]{
                {3},
                {1}};

        // when
        double result[][] = NNMath.matrixMultiply(a, b);

        // then
        String expected = "[[9.0 ]\n" +
                "[17.0 ]\n" +
                "[25.0 ]\n" +
                "[33.0 ]\n" +
                "]\n";

        System.out.println(MatrixUtil.matrixToString(result));
        assertThat(MatrixUtil.matrixToString(result), is(expected));
    }


    @Test
    public void shouldMultiply2x2into2x2() {
        // given
        double a[][] = new double[][]{
                {1, 0},
                {0, 1}};

        double b[][] = new double[][]{
                {4, 1},
                {2, 2}};

        // when
        double result[][] = NNMath.matrixMultiply(a, b);

        // then
        String expected = "[[4.0 1.0 ]\n" +
                "[2.0 2.0 ]\n" +
                "]\n";

        System.out.println(MatrixUtil.matrixToString(result));
        assertThat(MatrixUtil.matrixToString(result), is(expected));
    }

    @Test
    public void shouldMultiplyVectorsWithSameSize() throws Exception {
        double a[][] = new double[][]{
                {1, 5},
                {-2, 1}};

        double b[][] = new double[][]{
                {4, 1},
                {2, 2}};

        double[][] products = NNMath.scalarMultiply(a, b);

        String expected = "[[4.0 5.0 ]\n[-4.0 2.0 ]\n]\n";
        assertThat(MatrixUtil.matrixToString(products), is(expected));
    }


    @Test
    public void transposeMatrix() throws Exception {
        double a[][] = new double[][]{
                {1, 5},
                {-2, 3},
                {7, 9}};

        double[][] transposed = NNMath.matrixTranspose(a);

        String expected = "[[1.0 -2.0 7.0 ]\n[5.0 3.0 9.0 ]\n]\n";
        assertThat(MatrixUtil.matrixToString(transposed), is(expected));

    }

    @Test
    public void normalize() throws Exception {

        //given
        double[] input = {5,6,9};

        double[] normalized = NNMath.normalize(input);


        assertThat(normalized[0], is(0.25));
        assertThat(normalized[1], is(0.3));
        assertThat(normalized[2], is(0.45));


    }
}
