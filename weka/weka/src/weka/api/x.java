package weka.api;

import java.util.Scanner;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

public class x {
	 public static void main(String[] args) throws Exception {
		 
	            DataSource source = new DataSource("C:\\Users\\yazan\\eclipse-workspace\\weka\\src\\Height_Weight.csv");
	            Instances data = source.getDataSet();
	            int heightIndex = 1;
	            int weightIndex = 2;
	            AttributeStats heightStats = data.attributeStats(heightIndex);
	            AttributeStats weightStats = data.attributeStats(weightIndex);
	            System.out.println("Height Statistics:");
	            System.out.println("Min: " + heightStats.numericStats.min);
	            System.out.println("Max: " + heightStats.numericStats.max);
	            System.out.println("Mean: " + heightStats.numericStats.mean);
	            System.out.println("Standard Deviation: " + heightStats.numericStats.stdDev);
	            System.out.println("Median: " + getMedian(data, heightIndex));
	            System.out.println("\nWeight Statistics:");
	            System.out.println("Min: " + weightStats.numericStats.min);
	            System.out.println("Max: " + weightStats.numericStats.max);
	            System.out.println("Mean: " + weightStats.numericStats.mean);
	            System.out.println("Standard Deviation: " + weightStats.numericStats.stdDev);
	            System.out.println("Median: " + getMedian(data, weightIndex));
	        
		 
	//	CSVtoARFFConverter.Convert();
		
		int instances;
		 Scanner input =new Scanner (System.in);
		 System.out.println("pleas Enter the Number of Instances ");
		 instances=input.nextInt();
			LinearRegression(instances,source);

	}
	 public static void LinearRegression(int instances,DataSource source) {
	        try {
	            Instances data = source.getDataSet();

	            if (data != null) {
	                data.setClassIndex(data.numAttributes() - 1);
	                Randomize randomize = new Randomize();
	                randomize.setInputFormat(data);
	                data = Filter.useFilter(data, randomize);
	                Instances limitedData = new Instances(data, 0, instances);
	                int trainSize = (int) Math.round(limitedData.numInstances() * 0.7);
	                int testSize = limitedData.numInstances() - trainSize;
	                Instances trainData = new Instances(limitedData, 0, trainSize);
	                Instances testData = new Instances(limitedData, trainSize, testSize);
	                LinearRegression model = new LinearRegression();
	                model.buildClassifier(trainData);
	                Evaluation eval = new Evaluation(trainData);
	                eval.evaluateModel(model, testData);
	                System.out.println("Mean Absolute Error of M1: " + eval.meanAbsoluteError());
	                System.out.println("Root Mean Square Error of M1: "+ eval.rootMeanSquaredError());
	            } else {
	                System.err.println("Failed to load data from ARFF.");
	            }

	        } catch (Exception e) {
	            e.printStackTrace();
	        }
	    }
	 
	 private static double getMedian(Instances data, int columnIndex) {
	        int size = data.numInstances();
	        if (size % 2 == 0) {
	            double value1 = data.instance(size / 2 - 1).value(columnIndex);
	            double value2 = data.instance(size / 2).value(columnIndex);
	            return (value1 + value2) / 2.0;
	        } else {
	            return data.instance(size / 2).value(columnIndex);
	        }
	    }
}