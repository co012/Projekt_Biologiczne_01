package thirdtake;

import org.ejml.simple.SimpleMatrix;

public class Launcher {

    private final static double X0 = 0;
    private final static double Y0 = 0;
    public final static double L1 = 5;
    public final static double L2 = 5;
    public final static int LEARNING_DATA_CHUNK_NUMBER = 10000;
    public final static int LEARNING_DATA_CHUNK_LENGTH = 100;
    public final static int EPOCH = 1000;

    private final static int TEST_DATA_LENGTH = 100;


    private static SimpleMatrix getPoint(double alpha, double beta) {
        double gamma = 2 * Math.PI - alpha - beta;
        double x = X0;
        x += L1 * Math.sin(alpha);
        x += L2 * Math.sin(gamma);

        double y = Y0;
        y += L1 * Math.cos(alpha);
        y -= L2 * Math.cos(gamma);

        return new SimpleMatrix(2,1,true,new double[]{x, y});

    }

    private static SimpleMatrix[] generateOutputs(int number) {
        SimpleMatrix[] outputs = new SimpleMatrix[number];
        for(int i =0; i< number;i++) {
            outputs[i] = new SimpleMatrix(2,1);
            outputs[i].set(0,0,Math.random() * 2 * Math.PI);
            outputs[i].set(1,0,Math.random() * Math.PI);
        }
        return outputs;
    }

    public static void main(String[] args){
        NeuralNetwork neuralNetwork = new NeuralNetwork(
                new int[]{2,32,32,2},
                new ActivationFunction[]{ActivationFunction.RELU,ActivationFunction.RELU,ActivationFunction.ID}
                );

        final int LEARNING_DATA_LENGTH = LEARNING_DATA_CHUNK_LENGTH * LEARNING_DATA_CHUNK_NUMBER;
        SimpleMatrix[] outputs = generateOutputs(LEARNING_DATA_LENGTH);
        SimpleMatrix[] inputs = new SimpleMatrix[LEARNING_DATA_LENGTH];
        for (int i = 0; i < LEARNING_DATA_LENGTH; i++) {
            inputs[i] = getPoint(outputs[i].get(0),outputs[i].get(1));
        }

        for (int i = 0; i < EPOCH; i++) {
            for (int j = 0; j < LEARNING_DATA_CHUNK_NUMBER; j++) {
                for (int k = 0; k < LEARNING_DATA_CHUNK_LENGTH; k++) {
                    int index = LEARNING_DATA_CHUNK_LENGTH * j + k;
                    neuralNetwork.learn(inputs[index],outputs[index]);
                }
                neuralNetwork.getSmarter();
            }

            System.out.println("EPOCH " + i);
            System.out.println("Training MSE " + neuralNetwork.popSquareError()/TEST_DATA_LENGTH);
            System.out.println("New data MSE " + getNewDataMSE(neuralNetwork));
            System.out.println("--------------------------");
        }




    }

    private static double getNewDataMSE(NeuralNetwork neuralNetwork){
        double meanSquareError = 0;
        for (int i = 0; i < TEST_DATA_LENGTH; i++) {
            SimpleMatrix output = generateOutputs(1)[0];
            SimpleMatrix input = getPoint(output.get(0),output.get(1));
            SimpleMatrix output_neural = neuralNetwork.answer(input);
            SimpleMatrix diff = output.minus(output_neural);
            meanSquareError += diff.dot(diff);
        }
        meanSquareError/= TEST_DATA_LENGTH;

        SimpleMatrix output = generateOutputs(1)[0];
        SimpleMatrix input = getPoint(output.get(0),output.get(1));
        SimpleMatrix output_neural = neuralNetwork.answer(input);
        System.out.println("TEST:");
        System.out.println(input);
        System.out.println("Output: " + output.get(0) + ", " + output.get(1));
        System.out.println("Output predicted: " + output_neural.get(0) + ", " + output_neural.get(1));
        System.out.println("TEST END");


        return meanSquareError;
    }
}
