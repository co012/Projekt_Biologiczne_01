import org.ejml.simple.SimpleMatrix;

import java.util.LinkedList;

public class NeuralNetwork {
    private final LinkedList<Layer> layers;
    private double sumSquareError;
    private double learningRate = 1e-5;

    public NeuralNetwork(int[] outputs, ActivationFunction[] activationFunctions) {
        layers = new LinkedList<>();
        for (int i = 1; i < outputs.length; i++) {
            layers.add(new Layer(outputs[i - 1], outputs[i], activationFunctions[i - 1]));
        }
        sumSquareError = 0;

    }

    public SimpleMatrix answer(SimpleMatrix input) {
        for (Layer layer : layers) {
            input = layer.answer(input);
        }
        return input;
    }

    public void learn(SimpleMatrix input, SimpleMatrix output) {
        SimpleMatrix output_neural = answer(input);
        SimpleMatrix error = output.minus(output_neural);
        sumSquareError = error.dot(error);
        backPropagate(error.scale(-2 * learningRate));


    }

    public void getSmarter() {
        for (Layer layer : layers) {
            layer.getSmarter();
        }
    }

    private void backPropagate(SimpleMatrix grads) {
        var rIterator = layers.descendingIterator();
        while (rIterator.hasNext()) {
            Layer layer = rIterator.next();
            grads = layer.learn(grads);
        }
    }

}
