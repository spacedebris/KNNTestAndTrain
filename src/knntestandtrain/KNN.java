/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package knntestandtrain;

import java.io.FileNotFoundException;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
 *
 * @author si
 */
public class KNN {
    public static int k = -1;
    public static String wynik = "";
    public static String globalPath = "";
    static public void trainAndTestKNN() throws FileNotFoundException, IOException, Exception{
        globalPath = ChooseFile.globalPath;
        Instances trainData = DataLoad.loadData(globalPath);
        trainData.setClassIndex(trainData.numAttributes() - 1);

        Instances testData = DataLoad.loadData(globalPath);
        testData.setClassIndex(testData.numAttributes() - 1);

        IBk ibk = new IBk(k); 

        //Ustawienie odleglosci
        EuclideanDistance distance = new EuclideanDistance(); //euklidesowej
        //ManhattanDistance distance =  new ManhattanDistance(); //miejska              
        LinearNNSearch linearNN = new LinearNNSearch();        
        linearNN.setDistanceFunction(distance); //Ustwaienie odleglosci
        ibk.setNearestNeighbourSearchAlgorithm(linearNN); //ustawienie sposobu szukania sasiadow

        //Tworzenie klasyfikatora
        ibk.buildClassifier(trainData);

        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(ibk, testData);
        System.out.println("k = " +k);
        System.out.println(eval.toSummaryString("Wyniki:", false));
        
        wynik = eval.toSummaryString("Wyniki:", false);
    }
    
}
