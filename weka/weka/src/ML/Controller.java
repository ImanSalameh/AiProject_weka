package ML;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class Controller {
	public static void main(String[] args) {
		ML();
	}

	public static void initiateConversion() {
		try {
			loadCSVAndProcess();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Conversion Successful");
	}

	private static void loadCSVAndProcess() throws IOException {
		CSVLoader csvLoader = new CSVLoader();
		setCSVSourcePath(csvLoader);
		verifyCSVStructure(csvLoader);
		Instances dataset = csvLoader.getDataSet();
		processDataset(dataset);
	}

	private static void processDataset(Instances dataset) throws IOException {
		if (dataset != null) {
			for (int i = 0; i < dataset.numInstances(); i++) {
				updateDatasetValues(dataset, i);
			}
			saveAsARFF(dataset);
		} else {
			System.err.println("CSV Data Loading Failed.");
		}
	}

	private static void updateDatasetValues(Instances dataset, int index) throws IOException {
		String gender = dataset.instance(index).stringValue(0);
		double genderNumeric = gender.equalsIgnoreCase("Male") ? 1.0 : 0.0;
		dataset.instance(index).setValue(0, genderNumeric);
		double height = dataset.instance(index).value(1);
		dataset.instance(index).setValue(1, height * 2.54);
		double weight = dataset.instance(index).value(2);
		dataset.instance(index).setValue(2, weight * 0.453592);
	}

	private static void saveAsARFF(Instances dataset) throws IOException {
		ArffSaver arffSaver = new ArffSaver();
		arffSaver.setInstances(dataset);
		arffSaver.setFile(new File("C:\\Users\\yazan\\eclipse-workspace\\weka\\src\\output.arff"));
		arffSaver.writeBatch();
	}

	private static void setCSVSourcePath(CSVLoader csvLoader) throws IOException {
		csvLoader.setSource(new File("C:\\Users\\yazan\\eclipse-workspace\\weka\\src\\Height_Weight.csv"));
	}

	private static void verifyCSVStructure(CSVLoader csvLoader) throws IOException {
		if (csvLoader.getStructure() == null) {
			throw new IOException("Invalid CSV Structure.");
		}
	}

	private static void ML() {
		initiateConversion();
		DataStatistics.calculateAndDisplayStats();
		Module_ML1.performLinearRegression();
		Module_ML2.executeLinearRegression();
		Module_ML3.runLinearRegressionAnalysis();
		Module_ML4.executeLinearRegression();
	}
}
