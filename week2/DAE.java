package dl;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Log;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class DAE {
	
	private INDArray m_Weight;
	private INDArray m_Hbias;
	private INDArray m_Vbias;
	
	private int n_visible;
	private int n_hidden;
	
	public DAE(int n_visible, int n_hidden){
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;
		m_Weight = Nd4j.rand(new int[]{n_visible, n_hidden}, 1).mul(0.001);
		m_Hbias = Nd4j.rand(new int[]{1, n_hidden}, 1).mul(0.001);
		m_Vbias = Nd4j.rand(new int[]{1, n_visible}, 1).mul(0.001);
	}
	
	public INDArray encode(INDArray x){
		INDArray yt = x.mmul(m_Weight).addRowVector(m_Hbias);
		INDArray y = Nd4j.getExecutioner().execAndReturn(new Sigmoid(yt.dup()));
		return y;
	}
	
	public INDArray decode(INDArray y){
		INDArray zt = y.mmul(m_Weight.transpose()).addiRowVector(m_Vbias);
		INDArray z = Nd4j.getExecutioner().execAndReturn(new Sigmoid(zt.dup()));
		return z;
	}
	
	private INDArray get_corrupt_input(INDArray x, double corrupt_level){
		int n = x.columns();
		Random rng = new Random(1);
		double[] mask = new double[n];
		for(int i=0; i<n; i++){
			if(rng.nextDouble() > corrupt_level){
				mask[i] = 1;
			}
		}
		INDArray maskArray = Nd4j.create(mask);
		INDArray corrupt_x = x.mulRowVector(maskArray);
		return corrupt_x;
	}
	
	public void miniBatch(INDArray train_x, double learning_rate, double corrupt_level, int batch_size){
		int n = train_x.rows();
		int num = n/batch_size;
		INDArray temp_x;
		for(int i=0; i<num+1; i++){
			if(i != num){
				temp_x = train_x.get(NDArrayIndex.interval(i*batch_size,(i+1)*batch_size),NDArrayIndex.all());
			}else{
				temp_x = train_x.get(NDArrayIndex.interval(i*batch_size,n),NDArrayIndex.all());
			}
			train(temp_x, learning_rate, corrupt_level);
		}
	}
	
	public void train(INDArray train_x, double learning_rate, double corrupt_level){
		INDArray tilde_x = get_corrupt_input(train_x, corrupt_level);
		INDArray y = encode(tilde_x);
		INDArray z = decode(y);
		
		int n = train_x.rows();
		
		INDArray delta_v = train_x.sub(z);
		INDArray delta_h = delta_v.mmul(m_Weight).mul(y).mul(y.sub(1)).mul(-1);// y*(1-y) = y * (y-1) * -1
		INDArray delta_w = delta_v.transpose().mmul(y).add(tilde_x.transpose().mmul(delta_h));
		delta_w.divi(n);
		
		INDArray v_G = delta_v.sum(0).div(n);
		Nd4j.getBlasWrapper().level1().axpy(v_G.length(), learning_rate, v_G, m_Vbias);
		INDArray h_G = delta_h.sum(0).div(n);
		Nd4j.getBlasWrapper().level1().axpy(h_G.length(), learning_rate, h_G, m_Hbias);
		Nd4j.getBlasWrapper().level1().axpy(delta_w.length(), learning_rate, delta_w, m_Weight);
		
	}
	
	public INDArray reconstruct(INDArray test_x){
		INDArray y = encode(test_x);
		INDArray z = decode(y);
		return z;
	}
	
	public double reconstruct_error(INDArray test_x){
		INDArray z = reconstruct(test_x);
		INDArray logz = Nd4j.getExecutioner().execAndReturn(new Log(z.dup()));
		INDArray z_i = z.sub(1).mul(-1);
		INDArray log_zi = Nd4j.getExecutioner().execAndReturn(new Log(z_i.dup()));
		INDArray x = test_x.dup();
		INDArray x_i = x.sub(1).mul(-1);
		double error = x.mul(logz).add(x_i.mul(log_zi)).sumNumber().doubleValue();
		return error;
	}
	
	public void setWeight(INDArray weight){
		this.m_Weight = weight;
	}
	
	public INDArray getWeight(){
		return this.m_Weight;
	}
	
	public void setHbias(INDArray hbias){
		this.m_Hbias = hbias;
	}
	
	public INDArray getHbias(){
		return this.m_Hbias;
	}
	
	public void setVbias(INDArray vbias){
		this.m_Vbias = vbias;
	}
	
	public INDArray getVbias(){
		return this.m_Vbias;
	}
	
	public int getVisible(){
		return this.n_visible;
	}
	
	public int getHidden(){
		return this.n_hidden;
	}
	
	public static void main(String[] args){
		double learning_rate = 0.001;
		double corrupt_level = 0.3;
		int epochs = 10000;
		
		INDArray x = Nd4j.create(new double[][]{
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1}
		});
		
		DAE model = new DAE(20, 10);
		
		for(int i=0; i<epochs; i++){
			model.train(x, learning_rate, corrupt_level);
		}
		
		INDArray test_x = Nd4j.create(new double[][]{
			{1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1}
		});
		
		INDArray encode = model.encode(test_x);
		System.out.println(encode);
		INDArray recon = model.reconstruct(test_x);
		System.out.println(recon);
	}
}
