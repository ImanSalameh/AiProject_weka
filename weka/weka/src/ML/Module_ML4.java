package ML;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Module_ML4 {

	public static void executeLinearRegression() {
		try {
			Instances dataset = retrieveDataset("C:\\Users\\iman\\eclipse-workspace\\weka\\src\\Height_Weight.arff");
			if (dataset != null) {
				Instances shuffledData = shuffleDataset(dataset);
				Instances[] trainingAndTestingSets = partitionDataset(shuffledData, 0.7);
				Instances trainingSet = trainingAndTestingSets[0];
				Instances testingSet = trainingAndTestingSets[1];

				LinearRegression trainedModel = developRegressionModel(trainingSet);
				analyzeModelPerformance(trainedModel, trainingSet, testingSet);
			} else {
				System.err.println("ARFF  Loading Data ARFF Error.");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
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

	private static Instances retrieveDataset(String filePath) throws Exception {
		DataSource source = new DataSource(filePath);
		return source.getDataSet();
	}

	private static Instances shuffleDataset(Instances originalData) throws Exception {
		originalData.setClassIndex(originalData.numAttributes() - 1);
		Randomize randomizationFilter = new Randomize();
		randomizationFilter.setInputFormat(originalData);
		return Filter.useFilter(originalData, randomizationFilter);
	}

	private static Instances[] partitionDataset(Instances data, double trainingRatio) {
		int trainingSize = (int) Math.round(data.numInstances() * trainingRatio);
		Instances trainDataSet = new Instances(data, 0, trainingSize);
		Instances testDataSet = new Instances(data, trainingSize, data.numInstances() - trainingSize);
		return new Instances[] { trainDataSet, testDataSet };
	}

	private static LinearRegression developRegressionModel(Instances trainingData) throws Exception {
		LinearRegression regression = new LinearRegression();
		regression.buildClassifier(trainingData);
		return regression;
	}

	private static void analyzeModelPerformance(LinearRegression model, Instances trainData, Instances testData)
			throws Exception {
		Evaluation modelEvaluation = new Evaluation(trainData);
		modelEvaluation.evaluateModel(model, testData);
		printEvaluationTable(modelEvaluation);
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
