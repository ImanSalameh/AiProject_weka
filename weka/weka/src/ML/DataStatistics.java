package ML;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.AttributeStats;

public class DataStatistics {

	public static void calculateAndDisplayStats() {
		try {
			DataSource dataSource = initializeDataSource();
			Instances dataset = dataSource.getDataSet();
			AttributeStats heightStatistics = dataset.attributeStats(1);
			AttributeStats weightStatistics = dataset.attributeStats(2);

			printTableHeader();

			displayStatistics("Height", heightStatistics, dataset, 1);
			displayStatistics("Weight", weightStatistics, dataset, 2);
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

	private static DataSource initializeDataSource() throws Exception {
		return new DataSource("C:\\Users\\yazan\\eclipse-workspace\\weka\\src\\Height_Weight.arff");
	}

	private static void printTableHeader() {
		String outerBorder = "+=================================================================================+";
		String innerBorder = "|----------------+------+-------+------+---------------------+--------|";

		System.out.println(outerBorder);
		System.out.printf("| %-14s | %-4s | %-5s | %-4s | %-19s | %-6s |\n", "Attribute", "Min", "Max", "Mean",
				"Standard Deviation", "Median");
		System.out.println(innerBorder);
		System.out.println(outerBorder);
	}

	private static void displayStatistics(String attributeName, AttributeStats stats, Instances data,
			int attributeIndex) {
		String headerBorder = "+-----------+------+-------+------+--------------------+--------+";
		String header = "| Attribute | Min  | Max   | Mean | Standard Deviation | Median |";
		String separator = "|-----------|------|-------|------|--------------------|--------|";

		System.out.println(headerBorder);
		System.out.println(header);
		System.out.println(separator);

		System.out.printf("| %-9s | %-4.2f | %-5.2f | %-4.2f | %-18.2f | %-6.2f |\n", attributeName,
				stats.numericStats.min, stats.numericStats.max, stats.numericStats.mean, stats.numericStats.stdDev,
				calculateMedian(data, attributeIndex));

		System.out.println(headerBorder);
	}

	private static double calculateMedian(Instances data, int attributeIndex) {
		int dataSize = data.numInstances();
		if (dataSize % 2 == 0) {
			double midValue1 = data.instance(dataSize / 2 - 1).value(attributeIndex);
			double midValue2 = data.instance(dataSize / 2).value(attributeIndex);
			return (midValue1 + midValue2) / 2.0;
		} else {
			return data.instance(dataSize / 2).value(attributeIndex);
		}
	}
}
