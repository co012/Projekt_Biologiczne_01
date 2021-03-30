import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.LinkedList;

public class Launcher {
    private final static double X0 = 0;
    private final static double Y0 = 0;
    public final static double L1 = 10;
    public final static double L2 = 5;

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
            outputs[i].set(1,0,Math.random() * 2 * Math.PI);
        }
        return outputs;
    }

    public static void main(String[] args) {
        LinkedList<Integer> hiddenLayer = new LinkedList<>();
        hiddenLayer.add(4);
        hiddenLayer.add(4);
        hiddenLayer.add(2);
        NeuralNetworkMatrix neuralNetwork = new NeuralNetworkMatrix(2, hiddenLayer);
        final int learningDataLength = 100000;
        final int learningChunkLength = 5000;

        SimpleMatrix[] outputs = generateOutputs(learningDataLength);
        SimpleMatrix[] inputs = new SimpleMatrix[learningDataLength];
        for (int i = 0; i < learningDataLength; i++) {
            inputs[i] = getPoint(outputs[i].get(0), outputs[i].get(1));
        }
        double lastError = 0;

        for (int k = 0; k < 10000; k++) {


            for (int i = 0; i < learningDataLength / learningChunkLength; i++) {
                for (int j = 0; j < learningChunkLength; j++) {
                    int index = (learningChunkLength * i) + j;
                    neuralNetwork.learn(inputs[index], outputs[index]);
                }
                neuralNetwork.backPropagate();
            }
            System.out.print(outputs[0]);
            System.out.print(neuralNetwork.apply(inputs[0]));
            System.out.write('\n');
            double currError = neuralNetwork.popError()/learningDataLength;
            if(currError > lastError) NeuronMatrix.LEARNING_RATE *= 0.9;
            System.out.println("Epo: " + k  + " mean square error: " + currError);
            lastError = currError;

        }


    }
}
