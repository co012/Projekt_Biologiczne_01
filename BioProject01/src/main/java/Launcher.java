import java.util.Arrays;
import java.util.LinkedList;

public class Launcher {
    private final static double X0 = 0;
    private final static double Y0 = 0;
    private final static double L1 = 10;
    private final static double L2 = 5;

    private static double[] getPoint(double alpha, double beta) {
        double gamma = 2 * Math.PI - alpha - beta;
        double x = X0;
        x += L1 * Math.sin(alpha);
        x += L2 * Math.sin(gamma);

        double y = Y0;
        y += L1 * Math.cos(alpha);
        y -= L2 * Math.cos(gamma);

        return new double[]{x, y};

    }

    private static double[][] generateOutputs(int number) {
        double[][] outputs = new double[number][2];
        for (double[] output : outputs) {
            output[0] = Math.random() * 2 * Math.PI;
            output[1] = Math.random() * 2 * Math.PI;
        }
        return outputs;
    }

    public static void main(String[] args) {
        LinkedList<Integer> hiddenLayer = new LinkedList<>();
        hiddenLayer.add(64);
        hiddenLayer.add(64);
        hiddenLayer.add(64);
        hiddenLayer.add(2);
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, hiddenLayer);
        final int learningDataLength = 10000;
        final int learningChunkLength = 100;

        double[][] outputs = generateOutputs(learningDataLength);
        double[][] inputs = new double[learningDataLength][];
        for (int i = 0; i < learningDataLength; i++) {
            inputs[i] = getPoint(outputs[i][0], outputs[i][1]);
        }

        for (int k = 0; k < 10000; k++) {


            for (int i = 0; i < learningDataLength / learningChunkLength; i++) {
                for (int j = 0; j < learningChunkLength; j++) {
                    int index = (learningChunkLength * i) + j;
                    neuralNetwork.learn(inputs[index], outputs[index]);
                }
                neuralNetwork.backPropagate();
            }
            System.out.print(Arrays.toString(outputs[0]));
            System.out.print(Arrays.toString(neuralNetwork.apply(inputs[0])));
            System.out.write('\n');

        }


    }
}
