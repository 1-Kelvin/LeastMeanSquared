import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LeastMeanSquaredCalculator {

    private INDArray b_k;
    private INDArray p_k;
    private INDArray gamma_k;
    private float lambda;

    public LeastMeanSquaredCalculator(int dimension) {
        p_k = Nd4j.eye(dimension).mul(100);
        //b_k = Nd4j.zeros(2);
        b_k = Nd4j.create(new float[1 * 2], new int[]{1, 2});
        lambda = 0.99f;
    }

    private void update_b(INDArray x, INDArray y) {
        update_gamma(x);
        INDArray sub = y.sub(x.transpose().mmul(b_k));
        INDArray mmul = gamma_k.mmul(sub);
        b_k = b_k.add(mmul);
    }

    private void update_p (INDArray x) {
        INDArray inner = gamma_k.mmul(x.transpose().mmul(p_k));
        p_k = p_k.sub(inner).mul(1/lambda);
    }

    private void update_gamma(INDArray x) {
        INDArray counter = p_k.mmul(x);
        INDArray denominator = (x.transpose().mmul(p_k).mmul(x)).add(lambda);
        gamma_k = counter.div(denominator);
    }

    public INDArray calculate (INDArray x, INDArray y) {
        update_b(x, y);
        update_p(x);
        return b_k;
    }

    public INDArray getB_k() {
        return b_k;
    }

    public INDArray getP_k() {
        return p_k;
    }

    /*
    INDArray gamma = (P.mmul(x)).div(x.transpose().mmul(P).mmul(x).add(lambda));
    INDArray correction = y.sub(x.transpose().mul(b));

    INDArray b1=b.add(gamma.mul(correction));
    INDArray P1= P.sub(gamma.mul((x.transpose().mmul(P)))).mul(1/lambda );

    public INDArray addValue(INDArray b_k_plusOne) {
        if (b_k == null) {
            b_k = Nd4j.zeros(b_k_plusOne.shape());
            p_k = Nd4j.eye(b_k_plusOne.columns()).mul(100);
        }

        return null;
    }
    */
}
