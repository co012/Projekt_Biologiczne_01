public interface ActivationFunction {
    double getVal(double x);
    double getDerivativeVal(double x);

    ActivationFunction TANH = new ActivationFunction() {
        @Override
        public double getVal(double x) {
            return Math.tanh(x);
        }

        @Override
        public double getDerivativeVal(double x) {
            return 1 - Math.pow(Math.tanh(x),2);
        }
    };
}
