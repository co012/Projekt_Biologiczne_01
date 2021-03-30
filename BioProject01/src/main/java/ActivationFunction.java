public interface ActivationFunction {

    double getValue(double x);
    double getDerivativeValue(double x);

    ActivationFunction ID = new ActivationFunction() {
        @Override
        public double getValue(double x) {
            return x;
        }

        @Override
        public double getDerivativeValue(double x) {
            return 1;
        }
    };

    ActivationFunction TANH = new ActivationFunction() {
        @Override
        public double getValue(double x) {
            return Math.tanh(x);
        }

        @Override
        public double getDerivativeValue(double x) {
            return 1 - Math.pow(Math.tanh(x),2);
        }
    };
}
