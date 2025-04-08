package ML;

import weka.core.Instances;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

public class Module_ML3 {

	public static void runLinearRegressionAnalysis() {
		try {
			Instances dataset = loadDataset("C:\\Users\\iman\\eclipse-workspace\\weka\\src\\Height_Weight.arff");
			if (dataset != null) {
				Instances randomizedData = randomizeData(dataset);
				Instances[] splitDatasets = splitDataset(randomizedData, 5000, 0.7);
				Instances trainingDataset = splitDatasets[0];
				Instances testingDataset = splitDatasets[1];

				LinearRegression regression = createAndTrainModel(trainingDataset);
				evaluateRegressionModel(regression, trainingDataset, testingDataset);
			} else {
				System.err.println("Error loading ARFF data.");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static Instances loadDataset(String filePath) throws Exception {
		DataSource dataSource = new DataSource(filePath);
		return dataSource.getDataSet();
	}

	private static Instances randomizeData(Instances data) throws Exception {
		data.setClassIndex(data.numAttributes() - 1);
		Randomize randomizeFilter = new Randomize();
		randomizeFilter.setInputFormat(data);
		return Filter.useFilter(data, randomizeFilter);
	}

	private static Instances[] splitDataset(Instances data, int limit, double trainRatio) {
		Instances boundedData = new Instances(data, 0, limit);
		int trainSize = (int) Math.round(boundedData.numInstances() * trainRatio);
		Instances trainData = new Instances(boundedData, 0, trainSize);
		Instances testData = new Instances(boundedData, trainSize, boundedData.numInstances() - trainSize);
		return new Instances[] { trainData, testData };
	}

	private static LinearRegression createAndTrainModel(Instances trainData) throws Exception {
		LinearRegression model = new LinearRegression();
		model.buildClassifier(trainData);
		return model;
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

	private static void evaluateRegressionModel(LinearRegression model, Instances trainData, Instances testData)
			throws Exception {
		Evaluation evaluation = new Evaluation(trainData);
		evaluation.evaluateModel(model, testData);
		printEvaluationTable(evaluation);
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
