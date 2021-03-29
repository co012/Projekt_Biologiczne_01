import java.util.LinkedList;

public class Neuron {
    private double[] weights;
    private double[] delta_weights;
    private double bias;
    private double delta_bias;
    private double last_output;
    private double[] last_input;

    private static final double LEARNING_RATE = 1e-9;
    public final int inputsNumber;

    public Neuron(int inputs){
        inputsNumber = inputs;
        weights = new double[inputs];
        delta_weights = new double[inputs];
        for (int i = 0; i < inputs; i++) {
            weights[i] = Math.random();
            delta_weights[i] = 0;

        }
        bias = 0;
        delta_bias = 0;
    }


    public double apply(double[] input) {
        last_input = input;
        double result = 0;
        for (int i = 0; i < weights.length; i++) {
            result+= weights[i] * input[i];
        }

        last_output = result/inputsNumber+bias;
        return last_output;
    }

    public double[] backPropagate(double grad){
        double[] grads = new double[weights.length];
        for (int i = 0; i < grads.length; i++) {
            grads[i] = grad * weights[i];
        }

        for (int i = 0; i < delta_weights.length; i++) {
            delta_weights[i] += grad * last_input[i] * LEARNING_RATE;
        }

        return grads;
    }

    public void applyBackPropagation(){
        for (int i = 0; i < delta_weights.length; i++) {
            weights[i] -= delta_weights[i];
            delta_weights[i] = 0;
        }
    }
}
