package thirdtake;

import org.ejml.simple.SimpleBase;
import org.ejml.simple.SimpleMatrix;

public interface Optimizer {
    SimpleMatrix optimize(SimpleMatrix old,SimpleMatrix delta);
    Optimizer NONE = (old, delta) -> old;
    Optimizer SIMPLE = (SimpleBase::minus);

}
