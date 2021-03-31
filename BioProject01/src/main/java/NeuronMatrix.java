import org.ejml.dense.row.misc.ImplCommonOps_DDMA;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class NeuronMatrix {
    private SimpleMatrix weights;
    private SimpleMatrix delta_weights;
    private double bias;
    private double delta_bias;
    private double last_output;
    private SimpleMatrix last_input;

    public static double LEARNING_RATE = 1e-11;
    public final int inputsNumber;
    private final ActivationFunction activationFunction;


    public NeuronMatrix(int inputs, ActivationFunction activationFunction){
        this.activationFunction = activationFunction;
        inputsNumber = inputs;
        weights = SimpleMatrix.random_DDRM(inputsNumber,1,-1,1,new Random());
        delta_weights = new SimpleMatrix(inputsNumber,1);
        bias = 0;
        delta_bias = 0;
    }


    public double apply(SimpleMatrix input) {
        last_input = input;
        double result = weights.dot(input);
        last_output = activationFunction.getValue(result+bias);
        return last_output;
    }

    public SimpleMatrix backPropagate(double grad){
        SimpleMatrix grads = weights.scale(grad);

        delta_weights = last_input.scale(grad*LEARNING_RATE * activationFunction.getDerivativeValue(last_output)).plus(delta_weights);
        delta_bias += grad * LEARNING_RATE * activationFunction.getDerivativeValue(last_output);

        return grads;
    }

    public void applyBackPropagation(){
        weights = weights.minus(delta_weights);
        delta_weights.zero();
        bias -= delta_bias;
        delta_bias = 0;
    }
}
