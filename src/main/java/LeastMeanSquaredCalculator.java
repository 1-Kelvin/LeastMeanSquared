import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LeastMeanSquaredCalculator {

    private INDArray b_k;
    private INDArray p_k;
    private INDArray gamma_k;
    private float lambda;

    public LeastMeanSquaredCalculator(int dimension) {
        p_k = Nd4j.eye(dimension).mul(100);
        b_k = Nd4j.create(new float[][]{{0}, {0}});
        lambda = 1;
    }

    private void update_b(INDArray x, float y) {
        update_gamma(x);
        float sub = y - (x.transpose().mmul(b_k)).getFloat(0);
        INDArray mmul = gamma_k.mul(sub);
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

    public INDArray calculate (INDArray x, float y) {
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
}
