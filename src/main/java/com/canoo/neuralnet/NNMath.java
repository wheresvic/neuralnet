package com.canoo.neuralnet;

/**
 * Created by codecamp on 14/04/16.
 */
public class NNMath {

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public static double tanh(double x){
        return Math.tanh(x);
    }

    public static double tanhDerivative(double x){
        return 1 - Math.tanh(x) * Math.tanh(x);
    }

    public static double[][] matrixMultiply(double[][] a, double[][] b) {
        if (a.length == 0 || b.length == 0 || a[0].length != b.length) {
            throw new IllegalArgumentException("Cannot multiply non n x m and m x p matrices");
        }

        int n = a.length;
        int m = a[0].length;
        int p = b[0].length;

        double[][] result = new double[n][p];

        for (int nIter = 0; nIter < n; ++nIter) {
            for (int pIter = 0; pIter < p; ++pIter) {

                double sum = 0;
                for (int mIter = 0; mIter < m; ++mIter) {
                    sum += (a[nIter][mIter] * b[mIter][pIter]);
                }

                result[nIter][pIter] = sum;
            }
        }

        return result;
    }

    public static double[][] scalarMultiply(double[][] v1, double[][] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Cannot multiply vectors of unequal length");
        }
        double result[][] = new double[v1.length][v1[0].length];
        for (int i = 0; i < v1.length; ++i) {
            for (int j = 0; j < v1[i].length; ++j) {
                result[i][j] = v1[i][j] * v2[i][j];
            }
        }
        return result;

    }

    public static double[][] matrixSubtract(double[][] a, double[][] b) {
        if (a.length == 0 || b.length == 0 || a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Cannot subtract unequal matrices");
        }

        double result[][] = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[i].length; ++j) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }

        return result;
    }

    public static double[][] matrixAdd(double[][] a, double[][] b) {
        if (a.length == 0 || b.length == 0 || a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Cannot add unequal matrices");
        }

        double result[][] = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[i].length; ++j) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }

        return result;
    }

    public static double[][] matrixTranspose(double[][] matrix) {
        double[][] result = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; ++j) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    public static double[] normalize(double[] input){
        double sum = 0.0;
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            sum += input[i];
        }
        for (int i = 0; i < input.length; i++) {
            result[i] = input[i] / sum;
        }
        return result;
    }
}
