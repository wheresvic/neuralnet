package com.canoo.neuralnet;

import java.util.function.Function;

/**
 * Created by codecamp on 14/04/16.
 */
public class MatrixUtil {

    public static String matrixToString(double[][] matrix) {
        String result = "[";
        for (double[] aMatrix : matrix) {
            result += "[";
            for (double anAMatrix : aMatrix) {
                result += anAMatrix + " ";
            }
            result += "]\n";
        }
        result += "]\n";

        return result;
    }

    public static double[][] apply(double[][] matrix, Function<Double, Double> fn) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalArgumentException("Invalid matrix");
        }

        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                result[i][j] = fn.apply(matrix[i][j]);
            }
        }

        return result;
    }
}
