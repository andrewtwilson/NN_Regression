using System;
using System.Collections.Generic;

namespace Software_Sample
{
    class MainClass
    {
        /* In this software sample, a neural network (from Drew Wilson's NeuralNetwork library) 
         * is trained using training data generated from sin(x) with x range from 0 to "maxX"
         * and y range normalized between 0 and 1.
         * 
         * The software creates a "data.csv" file of the neural network's approximation of the 
         * sine function within the given range.
         */
        public static void Main(string[] args)
        {
            // initialize fields
            DataIO.DataExport DE = new DataIO.DataExport();
            MathClasses.Probability Prob = MathClasses.Probability.Instance;
            MathClasses.LinearAlgebra LA = new MathClasses.LinearAlgebra();
			NeuralNetworks.BackpropNetwork network = new NeuralNetworks.BackpropNetwork();

            int trainDataSize = 10000;
            int maxX = 10;
            int testDataSize = 10000;

			List<double[]> allInputs = new List<double[]>();
			List<double[]> allOutputs = new List<double[]>();
            double[] tempInput;
            double[] tempOutput;
			List<double[]> plotData = new List<double[]>();


            // create training data from sin(x) function
            for (int i = 0; i < trainDataSize; i++)
            {
                tempInput = new double[1];
                tempOutput = new double[1];
                tempInput[0] = Prob.NextDouble()*maxX;
                tempOutput[0] = Math.Sin(tempInput[0]);

                tempOutput[0] = (tempOutput[0] + 1) / 2; //normalize between 0 and 1

                allInputs.Add(tempInput);
                allOutputs.Add(tempOutput);
            }

			// set up network parameters (adjustable)
			network.NumInputs = 1;  // number of simulator inputs
			network.NumHidden = 5; // number of hidden nodes per network in simulator ensemble
			network.NumOutputs = 1; // number of outputs of simulator
			network.WeightInitSTD = .75; // std for random weight initialization
			network.Eta = 0.1; // learning rate for backprop
			network.Episodes = 2000; // episodes for backprop
			network.Momentum = 0.5; // momentum for backprop
			network.Shuffle = NeuralNetworks.ToggleShuffle.no; // do we shuffle training data during backprop?

            // give training data to network
			network.Inputs = allInputs;
			network.Outputs = allOutputs;
            network.ValidationInputs = new List<double[]>();
            network.ValidationOutputs = new List<double[]>();

            network.TrainNetwork();

            // test the network
			tempInput = new double[1];
			tempOutput = new double[1];
            for (int i = 0; i < testDataSize; i++)
            {
                tempInput[0] = (i+1) * (double)maxX / (double)testDataSize;
                tempOutput = network.NeuralNet.ForwardPass(tempInput);

                plotData.Add(LA.Join(tempInput,tempOutput));
            }
           
            DE.ExportData(plotData,"data");
        }
    }
}
