package com.krzem.cnn;
import java.util.ArrayList;
import java.util.List;



public class Main{
	public static void main(String[] args){
		List<Layer> ll=new ArrayList<Layer>();
		ll.add(new ConvolutionLayer(20,3,5,5,"sigmoid"));
		ll.add(new PoolingLayer(2,2));
		ll.add(new ConvolutionLayer(20,1,5,5,"sigmoid"));
		ll.add(new PoolingLayer(2,2));
		ll.add(new ConvolutionLayer(20,1,3,3,"sigmoid"));
		CNN cnn=new CNN(32,32,ll,300,1,new String[]{"airplanes","watch"},"sigmoid",7.5e-3);
		System.out.println(cnn.test(Dataset.load("./images/test",1)));
		System.out.println(cnn.classify(Dataset.get_img("./images/train/airplanes_0001.jpg")));
		cnn.train(Dataset.load("./images/train",2),Dataset.load("./images/test",2),100000,100000,true);
		System.out.println(cnn.test(Dataset.load("./images/train",1)));
		System.out.println(cnn.test(Dataset.load("./images/test",1)));
		System.out.println(cnn.classify(Dataset.get_img("./images/train/airplanes_0001.jpg")));
		System.out.println(cnn.classify(Dataset.get_img("./images/train/watch_0001.jpg")));
	}
}
