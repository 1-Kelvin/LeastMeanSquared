import org.nd4j.common.io.Assert;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

class LeastMeanSquaredCalculatorTest {

    @org.junit.jupiter.api.Test
    void calculate() {
        LeastMeanSquaredCalculator leastMeanSquaredCalculator = new LeastMeanSquaredCalculator(2);
        INDArray xvec0 = Nd4j.create(new float[][]{{1}, {0}});
        INDArray xvec1 = Nd4j.create(new float[][]{{1}, {1}});
        INDArray xvec2 = Nd4j.create(new float[][]{{1}, {2}});
        INDArray b_k;
        leastMeanSquaredCalculator.calculate(xvec0, 1);
        leastMeanSquaredCalculator.calculate(xvec1, 2);
        b_k = leastMeanSquaredCalculator.calculate(xvec2, 3);

        double k = b_k.getDouble(1);
        double d = b_k.getDouble(0);
        Assert.isTrue(Math.abs(1 - k) < 0.99 && Math.abs(1-d) < 0.99);
    }
}