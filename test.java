import com.krzem.cnn.*;
import java.util.ArrayList;
import java.util.List;



public class test{
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




























// import com.krzem.cnn.*;
// import java.util.ArrayList;
// import java.util.List;



// public class test{
// 	public static void main(String[] args){
// 		List<Layer> ll=new ArrayList<Layer>();
// 		ll.add(new ConvolutionLayer(5,5,4,20));
// 		ll.add(new PoolingLayer(2,2));
// 		ll.add(new ConvolutionLayer(5,5,1,20));
// 		ll.add(new PoolingLayer(2,2));
// 		ll.add(new ConvolutionLayer(3,3,1,20));
// 		ArrayList<String> cL=new ArrayList<String>();
// 		cL.add("airplanes");
// 		cL.add("butterfly");
// 		cL.add("flower");
// 		cL.add("grand_piano");
// 		cL.add("watch");
// 		CNN cnn=new CNN(32,32,ll,300,1,cL,"relu",0.01);
// 		System.out.println(cnn.classify(Dataset.get_img("./images/trainset/airplanes_0001.jpg")));
// 		cnn.train(Dataset.load("./images/trainset",4),Dataset.load("./images/tuneset",4),3,5,true);
// 		System.out.println(cnn.classify(Dataset.get_img("./images/trainset/airplanes_0001.jpg")));
// 	}
// }