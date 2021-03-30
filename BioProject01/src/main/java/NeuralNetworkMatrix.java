import org.ejml.simple.SimpleMatrix;

import java.util.LinkedList;
import java.util.List;

public class NeuralNetworkMatrix {

    private final LinkedList<NeuronMatrix[]> neuralNetworkNeurons;
    private double error;

    public NeuralNetworkMatrix(int inputNeuronNumber, List<Integer> hiddenLayersNeuronNumbers){

        neuralNetworkNeurons = new LinkedList<>();
        int previousInputs = inputNeuronNumber;
        for (Integer i : hiddenLayersNeuronNumbers){
            NeuronMatrix[] hiddenLayer = new NeuronMatrix[i];
            for (int j = 0; j < i; j++) {
                hiddenLayer[j] = new NeuronMatrix(previousInputs,hiddenLayersNeuronNumbers.get(hiddenLayersNeuronNumbers.size() - 1) == i ? ActivationFunction.ID:ActivationFunction.TANH);
            }
            previousInputs = i;
            neuralNetworkNeurons.add(hiddenLayer);
        }

    }


    public void learn(SimpleMatrix input, SimpleMatrix output) {
        SimpleMatrix output_neural = apply(input);
        error+= output.minus(output_neural).elementPower(2).elementSum();
        SimpleMatrix grads = output.minus(output_neural).scale(-2);
        var r_iterator = neuralNetworkNeurons.descendingIterator();
        while (r_iterator.hasNext()){
            NeuronMatrix[] neurons = r_iterator.next();
            SimpleMatrix grads_next = new SimpleMatrix(neurons[0].inputsNumber,1);
            for (int i = 0; i < neurons.length; i++) {
                grads_next = grads_next.plus(neurons[i].backPropagate(grads.get(i)));
            }
            grads = grads_next;
        }



    }

    public void backPropagate() {
        for (NeuronMatrix[] neurons : neuralNetworkNeurons){
            for( NeuronMatrix neuron : neurons){
                neuron.applyBackPropagation();
            }
        }

    }

    public SimpleMatrix apply(SimpleMatrix input) {


        var iterator = neuralNetworkNeurons.listIterator();
        SimpleMatrix output = new SimpleMatrix(0,0);
        while(iterator.hasNext()){
            NeuronMatrix[] layer = iterator.next();
            output = new SimpleMatrix(layer.length,1);
            for (int i = 0; i < layer.length; i++) {
                output.set(i,layer[i].apply(input));
            }
            input = output;
        }


        return output;
    }

    public double popError(){
        double e = error;
        error = 0;
        return e;
    }
}
