import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LeastMeanSquaredCalculator {

    private INDArray b_k;
    private INDArray p_k;

    public LeastMeanSquaredCalculator() {
    }

    public INDArray addValue(INDArray b_k_plusOne) {
        if (b_k == null) {
            b_k = Nd4j.zeros(b_k_plusOne.shape());
            p_k = Nd4j.eye(b_k_plusOne.columns()).mul(100);
        }

        return null;
    }

    /*
    INDArray gamma = (P.mmul(x)).div(x.transpose().mmul(P).mmul(x).add(lambda));
    INDArray correction = y.sub(x.transpose().mul(b));

    INDArray b1=b.add(gamma.mul(correction));
    INDArray P1= P.sub(gamma.mul((x.transpose().mmul(P)))).mul(1/lambda );
    */
}
