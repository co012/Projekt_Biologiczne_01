package thirdtake;

import org.ejml.simple.SimpleMatrix;

public class Launcher {

    private final static double X0 = 0;
    private final static double Y0 = 0;
    public final static double L1 = 15;
    public final static double L2 = 5;
    public final static int LEARNING_DATA_CHUNK_NUMBER = 100;
    public final static int LEARNING_DATA_CHUNK_LENGTH = 1000;
    public final static int EPOCH = 5000;

    private final static int TEST_DATA_LENGTH = 1000;


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
                new int[]{2,8,256,2},
                new ActivationFunction[]{ActivationFunction.LEAKY_RELU,ActivationFunction.LEAKY_RELU,ActivationFunction.ID}
                );

        final int LEARNING_DATA_LENGTH = LEARNING_DATA_CHUNK_LENGTH * LEARNING_DATA_CHUNK_NUMBER;


        for (int i = 0; i < EPOCH; i++) {
            SimpleMatrix[] outputs = generateOutputs(LEARNING_DATA_LENGTH);
            SimpleMatrix[] inputs = new SimpleMatrix[LEARNING_DATA_LENGTH];
            for (int a = 0; a < LEARNING_DATA_LENGTH; a++) {
                inputs[a] = getPoint(outputs[a].get(0),outputs[a].get(1));
            }

            for (int j = 0; j < LEARNING_DATA_CHUNK_NUMBER; j++) {
                for (int k = 0; k < LEARNING_DATA_CHUNK_LENGTH; k++) {
                    int index = LEARNING_DATA_CHUNK_LENGTH * j + k;
                    neuralNetwork.learn(inputs[index],outputs[index]);
                }
                neuralNetwork.getSmarter();
            }

            System.out.println("EPOCH " + i);
            System.out.println("Training MSE " + neuralNetwork.popSquareError()/LEARNING_DATA_LENGTH);
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
