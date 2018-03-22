/*
    Author: Johnathan M Melo Neto (jmmn.mg@gmail.com)
    Related paper: "On the Impact of the Objective Functions on Imbalanced Data Classification using Neuroevolution"

    This file is an adapted version of CGP-Library
    Copyright (c) Andrew James Turner 2014, 2015 (andrew.turner@york.ac.uk)
    The original CGP-Library is available in <http://www.cgplibrary.co.uk>    
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "cgpdelib.h"

double auc(struct parameters *, struct chromosome *, struct dataSet *, double threshold);
double gmean(struct parameters *, struct chromosome *, struct dataSet *, double threshold);
double fscore(struct parameters *, struct chromosome *, struct dataSet *, double threshold);
double accuracy(struct parameters *, struct chromosome *, struct dataSet *, double threshold);

int main(void)
{
    struct parameters *params = NULL;
    struct dataSet *mainData = NULL;

    // Insert the desired dataset here
    mainData = initialiseDataSetFromFile("./dataSets/diabetes.txt");

    // Initialize general parameters
    int numInputs = 8;  // attributes
    const int numOutputs = 1; // binary classification needs only 1 output node

    int numThreads = 10;
  
    int numNodes = 500;
    int nodeArity =  20;
    double weightRange = 5; 
    double mutationRate = 0.05;

    double CR = 0.90;
    double F = 0.70;

    // Set general parameters
    params = initialiseParameters(numInputs, numNodes, numOutputs, nodeArity);
    addNodeFunction(params, "sig");
    setMutationType(params, "probabilistic");
    setConnectionWeightRange(params, weightRange);
    setMutationRate(params, mutationRate);
    setNumThreads(params, numThreads);
    setCR(params, CR);
    setF(params, F);

    // Select one of the fitness functions: auc, gmean, fscore, accuracy
    setCustomFitnessFunction(params, auc, "AUC");
    //setCustomFitnessFunction(params, gmean, "Gmean");
    //setCustomFitnessFunction(params, fscore, "Fscore");
    //setCustomFitnessFunction(params, accuracy, "Accuracy");

    // Open text files to store the results
    FILE *f_CGP = fopen("./results/cgpann.txt", "w");
    FILE *f_IN = fopen("./results/cgpde_in.txt", "w");
    FILE *f_OUT = fopen("./results/cgpde_out.txt", "w");

    if (f_CGP == NULL || f_IN == NULL || f_OUT == NULL)
    {
        printf("Error opening files!\n");
        exit(1);
    }

    // CGPANN specific parameters
    int numGens_CGP = 50000;
	
    // CGPDE-IN specific parameters
    int numGens_IN = 64;
    int NP_IN = 10;
    int maxIter_IN = 400;

    setNP_IN(params, NP_IN);
    setMaxIter_IN(params, maxIter_IN);

    // CGPDE-OUT specific parameters
    int numGens_OUT = 40000;
    int NP_OUT = 20;
    int maxIter_OUT = 2570;

    setNP_OUT(params, NP_OUT);
    setMaxIter_OUT(params, maxIter_OUT);    

    // Header of the text files (to track the result of each independent run)
    fprintf(f_CGP, "i,\tj,\tauc,\tgmean,\tfscore,\taccuracy\n");
    fprintf(f_IN, "i,\tj,\tauc,\tgmean,\tfscore,\taccuracy\n");
    fprintf(f_OUT, "i,\tj,\tauc,\tgmean,\tfscore,\taccuracy\n");

    // Initialize the experiments
    printf("TYPE\t\ti\tj\tAUC\tGMEAN\tFSCORE\tACCURACY\n\n");
    int i, j;

    for(i = 0; i < 3; i++) // 3 independent cross-validations
    {
        // Set seed (for reproducibility purpose)
        unsigned int seed = i + 50;
        shuffleData(mainData, &seed);
        struct dataSet ** folds = generateFolds(mainData);

        #pragma omp parallel for default(none), private(j), shared(i,params,folds,numGens_CGP,numGens_IN,numGens_OUT,f_CGP,f_IN,f_OUT,NP_OUT), schedule(dynamic), num_threads(numThreads)
        for(j = 0; j < 10; j++) // stratified 10-fold cross-validation
        {
            // Set different seed for each independent run (for reproducibility purpose)
            unsigned int seed = (i*10)+j+5;

            // Build training, validation, and testing sets
            int * training_index = (int*)malloc(7*sizeof(int));
            int * validation_index = (int*)malloc(2*sizeof(int));
            int testing_index = j;
            getIndex(training_index, validation_index, testing_index, &seed);

            struct dataSet *trainingData = getTrainingData(folds, training_index);         
            struct dataSet *validationData = getValidationData(folds, validation_index);
            struct dataSet *testingData = getTestingData(folds, testing_index);

            /* Uncomment to save the training, validation, and testing sets 
            #pragma omp critical
            {
                char filename[100];
                char buf_i[10];
                char buf_j[10];
                memset(filename, '\0', sizeof(char)*100);
                memset(buf_i, '\0', sizeof(char)*10);
                memset(buf_j, '\0', sizeof(char)*10);
                snprintf(buf_i, 10, "%d", i);
                snprintf(buf_j, 10, "%d", j);
                strcat(filename, "./results/TRN/TRN_");
                strcat(filename, buf_i);
                strcat(filename, "_");
                strcat(filename, buf_j);
                strcat(filename, ".txt");
                saveDataSet(trainingData, filename);

                memset(filename, '\0', sizeof(char)*100);
                strcat(filename, "./results/VLD/VLD_");
                strcat(filename, buf_i);
                strcat(filename, "_");
                strcat(filename, buf_j);
                strcat(filename, ".txt");
                saveDataSet(validationData, filename);

                memset(filename, '\0', sizeof(char)*100);
                strcat(filename, "./results/TST/TST_");
                strcat(filename, buf_i);
                strcat(filename, "_");
                strcat(filename, buf_j);
                strcat(filename, ".txt");
                saveDataSet(testingData, filename);
            }*/

            /* 
                Run CGPANN 
            */
            struct chromosome * bestChromo = runCGP(params, trainingData, validationData, numGens_CGP, &seed);

            // GET CLASSIFICATION THRESHOLD
            double threshold = getThreshold(params, bestChromo, trainingData);

            // CGPANN TESTING AUC
            double CGPANN_AUC = auc(params, bestChromo, testingData, threshold);                

            // CGPANN TESTING GMEAN
            double CGPANN_GMEAN = gmean(params, bestChromo, testingData, threshold);

            // CGPANN TESTING FSCORE
            double CGPANN_FSCORE = fscore(params, bestChromo, testingData, threshold);

            // CGPANN TESTING ACCURACY
            double CGPANN_ACC = accuracy(params, bestChromo, testingData, threshold);

            printf("CGPANN\t\t%d\t%d\t%.4lf\t%.4lf\t%.4lf\t%.4lf\n", i, j, -CGPANN_AUC, -CGPANN_GMEAN, -CGPANN_FSCORE, -CGPANN_ACC);
            freeChromosome(bestChromo);  

            /*
                Run CGPDE-IN
            */
            bestChromo = runCGPDE_IN(params, trainingData, validationData, numGens_IN, &seed); 

            // GET CLASSIFICATION THRESHOLD
            threshold = getThreshold(params, bestChromo, trainingData);

            // CGPDE-IN TESTING AUC
            double CGPDE_IN_AUC = auc(params, bestChromo, testingData, threshold);

            // CGPDE-IN TESTING GMEAN
            double CGPDE_IN_GMEAN = gmean(params, bestChromo, testingData, threshold);

            // CGPDE-IN TESTING FSCORE
            double CGPDE_IN_FSCORE = fscore(params, bestChromo, testingData, threshold);

            // CGPDE-IN TESTING ACCURACY
            double CGPDE_IN_ACC = accuracy(params, bestChromo, testingData, threshold);

            printf("CGPDE-IN\t%d\t%d\t%.4lf\t%.4lf\t%.4lf\t%.4lf\n", i, j, -CGPDE_IN_AUC, -CGPDE_IN_GMEAN, -CGPDE_IN_FSCORE, -CGPDE_IN_ACC);
            freeChromosome(bestChromo);

            /*
                Run CGPDE-OUT 
            */
            struct chromosome ** populationChromos = runCGPDE_OUT(params, trainingData, validationData, numGens_OUT, &seed);    
            bestChromo = getBestDEChromosome(params, populationChromos, validationData, 3, &seed); //CGPDE-OUT-V: typeCGPDE = 3

            // GET CLASSIFICATION THRESHOLD
            threshold = getThreshold(params, bestChromo, trainingData);

            // CGPDE-OUT TESTING AUC
            double CGPDE_OUT_AUC = auc(params, bestChromo, testingData, threshold);

            // CGPDE-OUT TESTING GMEAN
            double CGPDE_OUT_GMEAN = gmean(params, bestChromo, testingData, threshold);

            // CGPDE-OUT TESTING FSCORE
            double CGPDE_OUT_FSCORE = fscore(params, bestChromo, testingData, threshold);

            // CGPDE-OUT TESTING ACCURACY
            double CGPDE_OUT_ACC = accuracy(params, bestChromo, testingData, threshold);

            printf("CGPDE-OUT\t%d\t%d\t%.4lf\t%.4lf\t%.4lf\t%.4lf\n", i, j, -CGPDE_OUT_AUC, -CGPDE_OUT_GMEAN, -CGPDE_OUT_FSCORE, -CGPDE_OUT_ACC);
            freeChromosome(bestChromo);           

            // Save the results
            #pragma omp critical
            {
                fprintf(f_CGP, "%d,\t%d,\t%.4f,\t%.4f,\t%.4f,\t%.4f\n", i, j, -CGPANN_AUC, -CGPANN_GMEAN, -CGPANN_FSCORE, -CGPANN_ACC);
                fprintf(f_IN, "%d,\t%d,\t%.4f,\t%.4f,\t%.4f,\t%.4f\n", i, j, -CGPDE_IN_AUC, -CGPDE_IN_GMEAN, -CGPDE_IN_FSCORE, -CGPDE_IN_ACC);
                fprintf(f_OUT, "%d,\t%d,\t%.4f,\t%.4f,\t%.4f,\t%.4f\n", i, j, -CGPDE_OUT_AUC, -CGPDE_OUT_GMEAN, -CGPDE_OUT_FSCORE, -CGPDE_OUT_ACC);
            }

            // Clear the chromosomes used by CGPDE-OUT version 
            int p;
            for (p = 0; p < NP_OUT; p++) 
            { 
                freeChromosome(populationChromos[p]);
            }
            free(populationChromos);

            // Clear training, validation, and testing sets
            freeDataSet(trainingData);
            freeDataSet(validationData);
            freeDataSet(testingData);
            free(training_index);
            free(validation_index);
        }

        // Clear folds
        int k;
        for(k = 0; k < 10; k++)
        {
            freeDataSet(folds[k]);
        }
        free(folds);

    }
	
    // Free the remaining variables
    freeDataSet(mainData);  
    freeParameters(params);
    fclose(f_CGP);
    fclose(f_IN);
    fclose(f_OUT);

    printf("\n* * * * * END * * * * *\n"); 

    return 0;
}

/* 
    F-score: measures the effectiveness of retrieval with respect to a user who attaches β 
        times as much importance to recall as precision
    Threshold: >= threshold (positive class) and < threshold (negative class)    
    Here, we aim to minimize -(fscore), which is equivalent to maximize +(fscore)
*/
double fscore(struct parameters *params, struct chromosome *chromo, struct dataSet *data, double threshold)
{
    int i;
    double beta = 1.0;

    if(getNumChromosomeInputs(chromo) != getNumDataSetInputs(data)){
        printf("Error: the number of chromosome inputs must match the number of inputs specified in the dataSet.\n");
        printf("Terminating.\n");
        exit(0);
    }

    if(getNumChromosomeOutputs(chromo) != 1){
        printf("Error: the number of chromosome outputs must be 1 in order to apply fscore for binary classification.\n");
        printf("Terminating.\n");
        exit(0);
    }

    int numDataSetSamples = getNumDataSetSamples(data);

    // store the output score of the predicted class for all samples
    double * outputs = (double *) malloc(numDataSetSamples * sizeof(double));

    // store the true labels of the samples
    double * labels = (double *) malloc(numDataSetSamples * sizeof(double));

    // build output and labels arrays & find the numbers of positive and negative instances
    unsigned long int total_negatives = 0, total_positives = 0;

    for(i = 0; i < numDataSetSamples; i++)
    {
        executeChromosome(chromo, getDataSetSampleInputs(data, i));        

        outputs[i] = getChromosomeOutput(chromo, 0);   

        if(getDataSetSampleOutput(data, i, 0) == 1.0) // (1,0): positive and (0,1): negative
        {
            labels[i] = 1.0; // minority / positive
            total_positives++;
        }
        else
        {
            labels[i] = 0.0; // majority / negative
            total_negatives++;
        }
    }

    // Create positive and negative groups
    int index_positive = 0, index_negative = 0;
    double * positive = (double *) malloc(total_positives * sizeof(double));
    double * negative = (double *) malloc(total_negatives * sizeof(double));

    for(i = 0; i < numDataSetSamples; i++)
    {
        if(labels[i] == 1.0) // POSITIVE
        {
            positive[index_positive] = outputs[i];
            index_positive++; 
        }
        else if(labels[i] == 0.0) // NEGATIVE
        {
            negative[index_negative] = outputs[i];
            index_negative++;
        }
    }

    // Calculate Recall (TPR)
    int correct_positives = 0;
    for(i = 0; i < total_positives; i++)
    {
        if(positive[i] >= threshold)
        {
            correct_positives++;
        }
    }

    if(correct_positives == 0)
    {
        // free memory
        free(outputs);
        free(labels);
        free(positive);
        free(negative);

        return -0.0;
    }
    
    double recall = (double)correct_positives / total_positives;

    // Calculate Precision
    int incorrect_positives = 0;
    for(i = 0; i < total_negatives; i++)
    {
        if(negative[i] >= threshold)
        {
            incorrect_positives++;
        }
    }

    double precision = (double)correct_positives / (correct_positives + incorrect_positives);

    // Calculate F-score
    double fscore = (double)( 1 + beta * beta ) * ( ( precision * recall ) / ( ( beta * beta * precision ) + recall ) ); 

    // free memory
    free(outputs);
    free(labels);
    free(positive);
    free(negative);
    
    return -fscore;
}

/* 
    G-mean: geometric mean between true positive rate and true negative rate
    Threshold: >= threshold (positive class) and < threshold (negative class)    
    Here, we aim to minimize -(gmean), which is equivalent to maximize +(gmean)
*/
double gmean(struct parameters *params, struct chromosome *chromo, struct dataSet *data, double threshold)
{
    int i;

    if(getNumChromosomeInputs(chromo) != getNumDataSetInputs(data)){
        printf("Error: the number of chromosome inputs must match the number of inputs specified in the dataSet.\n");
        printf("Terminating.\n");
        exit(0);
    }

    if(getNumChromosomeOutputs(chromo) != 1){
        printf("Error: the number of chromosome outputs must be 1 in order to apply gmean for binary classification.\n");
        printf("Terminating.\n");
        exit(0);
    }

    int numDataSetSamples = getNumDataSetSamples(data);

    // store the output score of the predicted class for all samples
    double * outputs = (double *) malloc(numDataSetSamples * sizeof(double));

    // store the true labels of the samples
    double * labels = (double *) malloc(numDataSetSamples * sizeof(double));

    // build output and labels arrays & find the numbers of positive and negative instances
    unsigned long int total_negatives = 0, total_positives = 0;

    for(i = 0; i < numDataSetSamples; i++)
    {
        executeChromosome(chromo, getDataSetSampleInputs(data, i));        

        outputs[i] = getChromosomeOutput(chromo, 0);           

        if(getDataSetSampleOutput(data, i, 0) == 1.0) // (1,0): positive and (0,1): negative
        {
            labels[i] = 1.0; // minority / positive
            total_positives++;
        }
        else
        {
            labels[i] = 0.0; // majority / negative
            total_negatives++;
        }
    }

    // Create positive and negative groups
    int index_positive = 0, index_negative = 0;
    double * positive = (double *) malloc(total_positives * sizeof(double));
    double * negative = (double *) malloc(total_negatives * sizeof(double));

    for(i = 0; i < numDataSetSamples; i++)
    {
        if(labels[i] == 1.0) // POSITIVE
        {
            positive[index_positive] = outputs[i];
            index_positive++; 
        }
        else if(labels[i] == 0.0) // NEGATIVE
        {
            negative[index_negative] = outputs[i];
            index_negative++;
        }
    }

    // Calculate True Positive Rate (TPR)
    int correct_positives = 0;
    for(i = 0; i < total_positives; i++)
    {
        if(positive[i] >= threshold)
        {
            correct_positives++;
        }
    }
    double TPR = (double)correct_positives / total_positives;

    // Calculate True Negative Rate (TNR)
    int correct_negatives = 0;
    for(i = 0; i < total_negatives; i++)
    {
        if(negative[i] < threshold)
        {
            correct_negatives++;
        }
    }
    double TNR = (double)correct_negatives / total_negatives;

    // Calculate G-mean
    double gmean = sqrt(TPR * TNR); 

    // free memory
    free(outputs);
    free(labels);
    free(positive);
    free(negative);

    return -gmean;
}

/* 
    AUC: Area Under the ROC Curve
    Here, we aim to minimize -(auc), which is equivalent to maximize +(auc)
    It does not need a threshold
*/
double auc(struct parameters *params, struct chromosome *chromo, struct dataSet *data, double threshold)
{
    int i;

    if(getNumChromosomeInputs(chromo) != getNumDataSetInputs(data)){
        printf("Error: the number of chromosome inputs must match the number of inputs specified in the dataSet.\n");
        printf("Terminating.\n");
        exit(0);
    }

    if(getNumChromosomeOutputs(chromo) != 1){
        printf("Error: the number of chromosome outputs must be 1 in order to apply auc for binary classification.\n");
        printf("Terminating.\n");
        exit(0);
    }

    int numDataSetSamples = getNumDataSetSamples(data);

    // store the output score of the predicted class for all samples
    double * outputs = (double *) malloc(numDataSetSamples * sizeof(double));

    // store the true labels of the samples
    double * labels = (double *) malloc(numDataSetSamples * sizeof(double));

    // build output and labels arrays
    unsigned long int majority_length = 0, minority_length = 0;

    for(i = 0; i < numDataSetSamples; i++)
    {
        executeChromosome(chromo, getDataSetSampleInputs(data, i));        

        outputs[i] = getChromosomeOutput(chromo, 0);           

        if(getDataSetSampleOutput(data, i, 0) == 1.0) // (1,0): positive and (0,1): negative
        {
            labels[i] = 1.0; // minority / positive
            minority_length++;
        }
        else
        {
            labels[i] = 0.0; // majority / negative
            majority_length++;
        }
    }

    double auc = calculateAuc(outputs, labels, numDataSetSamples, majority_length, minority_length);
    
    // free memory
    free(outputs);
    free(labels);

    return -auc;
}

/* 
    Accuracy: the proportion of correctly classified instances
    Threshold: >= threshold (positive class) and < threshold (negative class)    
    Here, we aim to minimize -(accuracy), which is equivalent to maximize +(accuracy)
*/
double accuracy(struct parameters *params, struct chromosome *chromo, struct dataSet *data, double threshold)
{
    int i;

    if(getNumChromosomeInputs(chromo) != getNumDataSetInputs(data))
    {
        printf("Error: the number of chromosome inputs must match the number of inputs specified in the dataSet.\n");
        printf("Terminating.\n");
        exit(0);
    }

    if(getNumChromosomeOutputs(chromo) != 1){
        printf("Error: the number of chromosome outputs must be 1 in order to apply accuracy for binary classification.\n");
        printf("Terminating.\n");
        exit(0);
    }

    int numDataSetSamples = getNumDataSetSamples(data);

    // store the output score of the predicted class for all samples
    double * outputs = (double *) malloc(numDataSetSamples * sizeof(double));

    // store the true labels of the samples
    double * labels = (double *) malloc(numDataSetSamples * sizeof(double));

    // build output and labels arrays & find the numbers of positive and negative instances
    unsigned long int total_negatives = 0, total_positives = 0;

    for(i = 0; i < numDataSetSamples; i++)
    {
        executeChromosome(chromo, getDataSetSampleInputs(data, i));        

        outputs[i] = getChromosomeOutput(chromo, 0);           

        if(getDataSetSampleOutput(data, i, 0) == 1.0) // (1,0): positive and (0,1): negative
        {
            labels[i] = 1.0; // minority / positive
            total_positives++;
        }
        else
        {
            labels[i] = 0.0; // majority / negative
            total_negatives++;
        }
    }

    // Create positive and negative groups
    int index_positive = 0, index_negative = 0;
    double * positive = (double *) malloc(total_positives * sizeof(double));
    double * negative = (double *) malloc(total_negatives * sizeof(double));

    for(i = 0; i < numDataSetSamples; i++)
    {
        if(labels[i] == 1.0) // POSITIVE
        {
            positive[index_positive] = outputs[i];
            index_positive++; 
        }
        else if(labels[i] == 0.0) // NEGATIVE
        {
            negative[index_negative] = outputs[i];
            index_negative++;
        }
    }

    // Calculate True Positives
    int correct_positives = 0;
    for(i = 0; i < total_positives; i++)
    {
        if(positive[i] >= threshold)
        {
            correct_positives++;
        }
    }

    // Calculate True Negatives
    int correct_negatives = 0;
    for(i = 0; i < total_negatives; i++)
    {
        if(negative[i] < threshold)
        {
            correct_negatives++;
        }
    }

    double accuracy = (double)(correct_positives + correct_negatives) / numDataSetSamples;

    // free memory
    free(outputs);
    free(labels);
    free(positive);
    free(negative);

    return -accuracy;
}