"""
This file is used to examine the dataset outputs.
The output is in allRespondentTypes.csv where you can see the following characteristics for the data:
- different types of respondents
- distribution of messages respondents
"""
from csv import reader
import collections
import csv
import matplotlib.pyplot as plt
FILE = "patientMessagesDataset5-18.csv"
RESULT = "allRespondentTypes.csv"
RESPONDENT_INDEX = 4

def main():
    file = open(FILE)
    next(file)
    all_respondent_types = set()

    def def_value():
        return 0
    distribution_counts = collections.defaultdict(def_value)
    total = 0

    # read file
    for line in reader(file):
        total += 1
        respondent = line[RESPONDENT_INDEX]
        all_respondent_types.add(respondent)
        distribution_counts[respondent] += 1

    # write results
    f = open(RESULT, 'w')
    writer = csv.writer(f)
    writer.writerow(["There are " + str(len(all_respondent_types)) + " unique outputs for categorization."])
    writer.writerow(list(all_respondent_types))
    writer.writerow(["respondent type", "percent of messages"])

    # calculate distributions, print chart and results
    labels = []
    percentages = []
    others_percentage = 0
    for respondent_type, count in distribution_counts.items():
        percentage = (count / total) * 100
        percentage = round(percentage, 2)
        writer.writerow([respondent_type, percentage])
        if percentage < 2:  # less than 2 percent, show in one "OTHERS" bucket
            others_percentage += percentage
        else:
            labels.append(respondent_type)
            percentages.append(percentage)
    # add the others in
    labels.append("OTHERS")
    percentages.append(others_percentage)
    # make pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

if __name__ == "__main__":
    main()