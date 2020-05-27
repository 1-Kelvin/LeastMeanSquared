import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
    public static void main(String[] args) {
        LeastMeanSquaredCalculator leastMeanSquaredCalculator = new LeastMeanSquaredCalculator(2);
        INDArray xvec0 = Nd4j.create(new float[][]{{1}, {0}});
        INDArray xvec1 = Nd4j.create(new float[][]{{1}, {1}});
        INDArray xvec2 = Nd4j.create(new float[][]{{1}, {2}});

        INDArray b_k;
        b_k = leastMeanSquaredCalculator.calculate(xvec0, 1);
        System.out.println(b_k.toStringFull());
        b_k = leastMeanSquaredCalculator.calculate(xvec1, 2);
        System.out.println(b_k.toStringFull());
        b_k = leastMeanSquaredCalculator.calculate(xvec2, 3);
        System.out.println(b_k.toStringFull());
    }
}
