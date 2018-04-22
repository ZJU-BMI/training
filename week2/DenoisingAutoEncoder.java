package deeplearning;

import static deeplearning.Util.*;

import java.util.Random;

/**
 * 
 * @author GZX-ZJU
 *
 */

public class DenoisingAutoEncoder {
	
	public int n_visible;
	public int n_hidden;
	
	public double[][] W;
	public double[] hbias;
	public double[] vbias;
	public Random rng;
	
	public DenoisingAutoEncoder(int n_visible, int n_hidden){
		this(n_visible, n_hidden, null,null,null,null);
	}
	
	public DenoisingAutoEncoder(int n_visible, int n_hidden, double[][] W, double[] hbias, double[] vbias, Random rng){
		this.n_hidden = n_hidden;
		this.n_visible = n_visible;
		
		if(rng == null) this.rng = new Random(1);
		
		if(W == null) {
			this.W = new double[n_visible][n_hidden];
			for(int i=0; i<n_visible; i++){
				for(int j=0; j<n_hidden; j++){
					this.W[i][j] = this.rng.nextDouble() / 1000;
				}
			}
		}
		else this.W = W;
		
		if(hbias == null) this.hbias = new double[n_hidden];
		else this.hbias = hbias;
		
		if(vbias == null) this.vbias = new double[n_visible];
		else this.vbias = vbias;
		
	}
	
	public double[] encode(double[] x){
		
		double[] y = new double[n_hidden];
		
		for(int i=0; i < n_hidden; i++){
			y[i] = 0;
			for(int j=0; j<n_visible; j++){
				y[i] += W[j][i] * x[j];
			}
			y[i] += hbias[i];
			y[i] = sigmoid(y[i]);
		}		
		
		return y;
	}
	
	public double[] decode(double[] y){
		
		double[] z = new double[n_visible];
		
		for(int i=0; i<n_visible; i++){
			z[i] = 0;
			for(int j=0; j<n_hidden; j++){
				z[i] += W[i][j] * y[j];
			}
			z[i] += vbias[i];
			z[i] = sigmoid(z[i]);
		}
		return z;
	}
	
	public double[] get_corrupt_input(double[] x, double corrupt_level){
		
		double[] tilde_x = new double[x.length];
		
		for(int i=0; i<x.length; i++){
			if(rng.nextDouble() >= corrupt_level){
				tilde_x[i] = x[i];
			}else{
				tilde_x[i] = 0;
			}
		}
		
		return tilde_x;
	}
	
	
	public void train(double[] train_x, double learning_rate, double corrupt_level){
		
		double[] tilde_x = new double[n_visible];
		double[] y = new double[n_hidden];
		double[] z = new double[n_visible];
		
		double[] h = new double[n_hidden];
		double[] v = new double[n_visible];
		
		tilde_x = get_corrupt_input(train_x, corrupt_level);
		y = encode(tilde_x);
		z = decode(y);
		
		//vbias
		for(int i=0; i<n_visible; i++){
			v[i] = train_x[i] - z[i];
			vbias[i] += v[i] * learning_rate;
		}
		
		//hbias
		for(int i=0; i<n_hidden; i++){
			h[i] = 0;
			for(int j=0; j<n_visible; j++){
				h[i] += W[j][i] * v[j];
			}
			h[i] *= y[i] * (1-y[i]);
			hbias[i] += h[i] * learning_rate;
		}
		
		//W
		for(int i=0; i<n_visible; i++){
			for(int j=0; j<n_hidden; j++){
				W[i][j] += (h[j] * tilde_x[i] + v[i] * y[j]) * learning_rate;
			}
		}
		
	}
	
	public double[] reconstruct(double[] x){
		 double[] y = encode(x);
		 double[] z = decode(y);
		 return z;
	}
	
	public double reconstruction_error(double[] x){
		double[] z = reconstruct(x);
		
		double error = 0;
		for(int i=0; i<n_visible; i++){
			error += x[i] * Math.log(z[i]) + (1-x[i]) * Math.log(1-z[i]);
		}
		return error;
	}
	
	public static void test_dA(){
		
		double learning_rate = 0.1;
		double corrupt_level = 0.3;
		int train_epochs = 1000;
		
		int n_visible = 20;
		int n_hidden = 5;
		
		int train_n = 10;
		double[][] train_x = {
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
			};
		
		DenoisingAutoEncoder dA = new DenoisingAutoEncoder(n_visible, n_hidden, null, null, null, null);
		
		for(int i = 0; i<train_epochs; i++){
			for(int j=0; j<train_n; j++){
				dA.train(train_x[j], learning_rate, corrupt_level);
			}
		}
		
		double[][] test_x = {
                {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
        };
		
		for(int i=0; i<2; i++){
			double[] code = dA.encode(test_x[i]);
			double[] ct = dA.reconstruct(test_x[i]);
			for(int j=0; j<n_visible; j++){
				System.out.printf("%.2f ", ct[j]);
			}
			System.out.println();
			for(int j=0; j<n_hidden; j++){
				System.out.printf("%.2f ", code[j]);				
			}
			System.out.println();
			System.out.println();
		}
		
	}
	
	public static void main(String[] args){
		test_dA();
	}
	
	
	
	
	
	
	
	
	
}
