import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class NeuralNetwork {
    private final LinkedList<Neuron[]> neuralNetworkNeurons;

    public NeuralNetwork(int inputNeuronNumber, List<Integer> hiddenLayersNeuronNumbers){

        neuralNetworkNeurons = new LinkedList<>();
        int previousInputs = inputNeuronNumber;
        for (Integer i : hiddenLayersNeuronNumbers){
            Neuron[] hiddenLayer = new Neuron[i];
            for (int j = 0; j < i; j++) {
                hiddenLayer[j] = new Neuron(previousInputs);
            }
            previousInputs = i;
            neuralNetworkNeurons.add(hiddenLayer);
        }

    }


    public void learn(double[] input, double[] output) {
        double[] output_neural = apply(input);
        double[] grads = new double[]{ -2 * (output[0] - output_neural[0]) };
        var r_iterator = neuralNetworkNeurons.descendingIterator();
        while (r_iterator.hasNext()){
            Neuron[] neurons = r_iterator.next();
            double[] grads_next = new double[neurons[0].inputsNumber];
            for (int i = 0; i < neurons.length; i++) {
                double[] grads_part = neurons[i].backPropagate(grads[i]);
                for (int j = 0; j < grads_part.length; j++) {
                    grads_next[j]+=grads_part[j];
                }
            }
            grads = grads_next;

        }



    }

    public void backPropagate() {
        for (Neuron[] neurons : neuralNetworkNeurons){
            for( Neuron neuron : neurons){
                neuron.applyBackPropagation();
            }
        }

    }

    public double[] apply(double[] input) {


        var iterator = neuralNetworkNeurons.listIterator();
        double[] output = null;
        while(iterator.hasNext()){
            Neuron[] layer = iterator.next();
            output = new double[layer.length];
            for (int i = 0; i < output.length; i++) {
                output[i] = layer[i].apply(input);
            }
            input = output;
        }


        return output;
    }
}
