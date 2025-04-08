package ML;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

public class Module_ML2 {

	public static void executeLinearRegression() {
		try {
			Instances dataset = fetchDataSet("C:\\Users\\iman\\eclipse-workspace\\weka\\src\\Height_Weight.arff");
			if (dataset != null) {
				Instances processedData = processDataForAnalysis(dataset);
				Instances[] dividedData = divideData(processedData);
				Instances trainingSet = dividedData[0];
				Instances testingSet = dividedData[1];
				LinearRegression lrModel = trainLinearRegressionModel(trainingSet);
				performEvaluation(lrModel, trainingSet, testingSet);
			} else {
				System.err.println("Data loading from ARFF file failed.");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static Instances fetchDataSet(String path) throws Exception {
		DataSource dataSource = new DataSource(path);
		return dataSource.getDataSet();
	}

	private static Instances processDataForAnalysis(Instances originalData) throws Exception {
		originalData.setClassIndex(originalData.numAttributes() - 1);
		Randomize randomizer = new Randomize();
		randomizer.setInputFormat(originalData);
		return Filter.useFilter(originalData, randomizer);
	}

	private static Instances[] divideData(Instances preparedData) {
		int sizeLimit = 500;
		Instances boundedData = new Instances(preparedData, 0, sizeLimit);
		int trainingSize = (int) Math.round(boundedData.numInstances() * 0.7);
		Instances trainingData = new Instances(boundedData, 0, trainingSize);
		Instances testingData = new Instances(boundedData, trainingSize, boundedData.numInstances() - trainingSize);
		return new Instances[] { trainingData, testingData };
	}

	private static LinearRegression trainLinearRegressionModel(Instances trainData) throws Exception {
		LinearRegression regressionModel = new LinearRegression();
		regressionModel.buildClassifier(trainData);
		return regressionModel;
	}

	private static void performEvaluation(LinearRegression model, Instances trainData, Instances testData)
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
