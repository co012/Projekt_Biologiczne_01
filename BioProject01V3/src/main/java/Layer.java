import org.ejml.simple.SimpleMatrix;

import java.util.Random;
import java.util.function.Function;

public class Layer {
    private SimpleMatrix weights;
    private SimpleMatrix delta_weights;
    private SimpleMatrix bias;
    private SimpleMatrix delta_bias;
    private SimpleMatrix last_input,last_output;

    private final ActivationFunction activationFunction;

    public Layer(int inputsNumber,int outputsNumber,ActivationFunction activationFunction){
        this.activationFunction = activationFunction;
        Random random = new Random();
        weights = SimpleMatrix.random_DDRM(outputsNumber,inputsNumber,-1,1,random);
        delta_weights = new SimpleMatrix(outputsNumber,inputsNumber);
        bias = SimpleMatrix.random_DDRM(outputsNumber,1,-1,1,random);
        delta_bias = new SimpleMatrix(outputsNumber,1);

    }

    public SimpleMatrix answer(SimpleMatrix input){
        last_input = input;
        SimpleMatrix output = weights.mult(input).plus(bias);
        applyFunction(output, activationFunction::getVal);
        last_output = output;
        return output;
    }

    public SimpleMatrix learn(SimpleMatrix grads){
        applyFunction(last_output,activationFunction::getDerivativeVal);
        grads.elementMult(last_output);
        delta_bias = delta_bias.plus(grads);
        delta_weights = last_input.transpose().kron(grads).plus(delta_weights);
        return weights.transpose().mult(grads);

    }

    public void getSmarter(){
        weights = weights.minus(delta_weights);
        delta_weights.zero();

        bias = bias.minus(delta_bias);
        delta_bias.zero();

    }

    private void applyFunction(SimpleMatrix s, Function<Double,Double> f){
        for (int i = 0; i < s.getNumElements(); i++) {
            s.set(i,f.apply(s.get(i)));
        }
    }



}
