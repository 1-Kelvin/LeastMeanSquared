import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
    public static void main(String[] args) {
        LeastMeanSquaredCalculator leastMeanSquaredCalculator = new LeastMeanSquaredCalculator(1);
        float[] x = {3};
        float[] y = {2};
        INDArray xvec = Nd4j.create(x, new int[]{1,1});
        INDArray yvec = Nd4j.create(y, new int[]{1,1});
        //leastMeanSquaredCalculator.calculate(xvec, yvec);
        INDArray b_k = leastMeanSquaredCalculator.calculate(xvec, yvec);
        //b_k = leastMeanSquaredCalculator.calculate(xvec, yvec);
        //b_k = leastMeanSquaredCalculator.calculate(xvec, yvec);
        //b_k = leastMeanSquaredCalculator.calculate(xvec, yvec);
        System.out.println(b_k.toStringFull());
        System.out.println(leastMeanSquaredCalculator.getP_k().toStringFull());
    }
}
