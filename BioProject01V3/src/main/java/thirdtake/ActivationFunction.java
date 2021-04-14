package thirdtake;

import javax.xml.namespace.QName;

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
            return 1 - Math.pow(Math.tanh(x), 2);
        }

        @Override
        public String toString() {
            return "TANH";
        }
    };

    ActivationFunction ID = new ActivationFunction() {
        @Override
        public double getVal(double x) {
            return x;
        }

        @Override
        public double getDerivativeVal(double x) {
            return 1;
        }

        @Override
        public String toString() {
            return "ID";
        }
    };

    ActivationFunction RELU = new ActivationFunction() {
        @Override
        public double getVal(double x) {
            return Math.max(0, x);
        }

        @Override
        public double getDerivativeVal(double x) {
            return x > 0 ? 1 : 0;
        }

        @Override
        public String toString() {
            return "RELU";
        }
    };

    ActivationFunction LEAKY_RELU = new ActivationFunction() {
        @Override
        public double getVal(double x) {
            return x > 0 ? x : 0.01 * x;
        }

        @Override
        public double getDerivativeVal(double x) {
            return x > 0 ? 1 : 0.01;
        }

        @Override
        public String toString() {
            return "LEAKY_RELU";
        }
    };

}
