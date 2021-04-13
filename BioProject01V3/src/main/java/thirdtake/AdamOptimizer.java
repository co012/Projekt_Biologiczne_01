package thirdtake;

import org.ejml.simple.SimpleMatrix;

public class AdamOptimizer implements Optimizer {

    private final static double alpha = 1e-3;
    private final static double beta_1 = 0.9;
    private final static double beta_2 = 0.999;
    private final static double epsilon = 1e-8;

    private SimpleMatrix m;
    private SimpleMatrix v;

    private int step;

    public AdamOptimizer(int rows,int cols){
        m = new SimpleMatrix(rows,cols);
        v = new SimpleMatrix(rows,cols);
        step = 0;
//        m = beta_1 * m + (1 - beta_1) * g
//        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
//        m_hat = m / (1 - np.power(beta_1, t))
//        v_hat = v / (1 - np.power(beta_2, t))
//        w = w - step_size * m_hat / (np.sqrt(v_hat) + epsilon)
    }


    @Override
    public SimpleMatrix optimize(SimpleMatrix old, SimpleMatrix delta) {
        step++;
        m = m.scale(beta_1).plus(delta.scale(1-beta_1));
        v = v.scale(beta_2).plus(delta.elementMult(delta).scale(1-beta_2));
        SimpleMatrix m_hat = m.divide(1 - Math.pow(beta_1,step));
        SimpleMatrix v_hat = v.divide(1 - Math.pow(beta_2,step));
        return m_hat.scale(alpha).elementDiv(v_hat.elementPower(0.5).plus(epsilon)).minus(old).scale(-1);

    }
}
