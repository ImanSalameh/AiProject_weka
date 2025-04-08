package ML;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

public class Module_ML1 {
	public static void performLinearRegression() {
		try {
			Instances data = loadData("C:\\Users\\iman\\eclipse-workspace\\weka\\src\\Height_Weight.arff");
			if (data != null) {
				data = prepareData(data);
				Instances[] splitData = splitData(data);
				Instances trainData = splitData[0];
				Instances testData = splitData[1];
				LinearRegression model = buildModel(trainData);
				evaluateModel(model, trainData, testData);
			} else {
				System.err.println("Failed to load data from ARFF.");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static Instances loadData(String filePath) throws Exception {
		DataSource source = new DataSource(filePath);
		return source.getDataSet();
	}

	private static Instances prepareData(Instances data) throws Exception {
		data.setClassIndex(data.numAttributes() - 1);
		Randomize randomize = new Randomize();
		randomize.setInputFormat(data);
		return Filter.useFilter(data, randomize);
	}

	private static Instances[] splitData(Instances data) {
		int instancesLimit = 100;
		Instances limitedData = new Instances(data, 0, instancesLimit);
		int trainSize = (int) Math.round(limitedData.numInstances() * 0.7);
		int testSize = limitedData.numInstances() - trainSize;
		Instances trainData = new Instances(limitedData, 0, trainSize);
		Instances testData = new Instances(limitedData, trainSize, testSize);
		return new Instances[] { trainData, testData };
	}

	private static LinearRegression buildModel(Instances trainData) throws Exception {
		LinearRegression model = new LinearRegression();
		model.buildClassifier(trainData);
		return model;
	}

	private static void evaluateModel(LinearRegression model, Instances trainData, Instances testData)
			throws Exception {
		Evaluation eval = new Evaluation(trainData);
		eval.evaluateModel(model, testData);
		printEvaluationTable(eval);
	}

	public static void printDataToFile(String filePath, String dataToPrint) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filePath));
			writer.write(dataToPrint);
			writer.close();
			System.out.println("Data has been successfully written to the file: " + filePath);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void printEvaluationTable(Evaluation eval) {
		String headerBorder = "+---------------------------+------------------+";
		String dataBorder = "+---------------------------+------------------+";

		System.out.println(headerBorder);
		System.out.printf("| %-25s | %-16s |\n", "Metric", "Value");
		System.out.println(headerBorder);

		System.out.printf("| %-25s | %-16.4f |\n", "Mean Absolute Error", eval.meanAbsoluteError());
		System.out.printf("| %-25s | %-16.4f |\n", "Root Mean Squared Error", eval.rootMeanSquaredError());

		System.out.println(dataBorder);
	}

}
