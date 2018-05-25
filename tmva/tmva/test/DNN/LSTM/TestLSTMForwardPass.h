// @(#)root/tmva $Id$
// Author: Harshit Prasad

/*************************************************************************
 * Copyright (C) 2018, Harshit Prasad                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Generic tests of the LSTM-Layer Forward Pass                   //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_LSTM_TEST_LSTM_FWDPASS_H
#define TMVA_TEST_DNN_TEST_LSTM_TEST_LSTM_FWDPASS_H

#include <iostream>
#include <vector>

#include "../Utility.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::LSTM;

//______________________________________________________________________________
/* Prints out Tensor, printTensor1(A, matrix) */
template <typename Architecture>
auto printTensor1(const std::vector<typename Architecture::Matrix_t> &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t l = 0; l < A.size(); ++l) {
      for (size_t i = 0; i < (size_t) A[l].GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) A[l].GetNcols(); ++j) {
            std::cout << A[l](i, j) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "********\n";
  } 
}

//______________________________________________________________________________
/* Prints out Matrix, printMatrix1(A, matrix) */
template <typename Architecture>
auto printMatrix1(const typename Architecture::Matrix_t &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t i = 0; i < (size_t) A.GetNrows(); ++i) {
    for (size_t j = 0; j < (size_t) A.GetNcols(); ++j) {
        std::cout << A(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "********\n";
}

double sigmoid(double x) { return 1 / 1 + exp(-x); }

/*! Generic sample test for forward propagation in LSTM network. */
//______________________________________________________________________________
template <typename Architecture>
auto testForwardPass(size_t timeSteps, size_t batchSize, size_t stateSize, size_t inputSize)
-> Double_t
{
  using Matrix_t = typename Architecture::Matrix_t;
  using Tensor_t = std::vector<Matrix_t>;
  using LSTMLayer_t = TBasicLSTMLayer<Architecture>;
  using Net_t = TDeepNet<Architecture>;

  //______________________________________________________________________________
  /* Input Gate: Numerical example. 
   * Reference: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9 
   * TODO: Numerical example for other gates to verify forward pass values and 
   * backward pass values. */

  /* Net_t LSTM(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
  LSTMLayer_t* layer = LSTM.AddBasicLSTMLayer(stateSize, inputSize, timeSteps);

  layer->Initialize();

  Matrix_t X(inputSize, stateSize); // inputSize = 1, stateSize = 2
  X(0,0) = 1; X(1,0) = 2;

  Matrix_t Wi(stateSize, inputSize); // stateSize = 2, inputSize = 1
  Wi(0,0) = 0.95; Wi(1,0) = 0.8;
  copyMatrix(layer->GetWeightsInputGate(), Wi);

  Matrix_t H(batchSize, stateSize); // batchSize = 2, stateSize = 2
  H(0,0) = 1.1; H(0,1) = 2.2; H(1,0) = 3.1; H(1,1) = 0.9;
  copyMatrix(layer->GetState(), H);

  Matrix_t Ws(stateSize, stateSize); // stateSize = 2, stateSize = 2
  Ws(0,0) = 0.52; Ws(0,1) = 0.3; Ws(1,0) = 0.11; Ws(1,1) = 0.8;
  copyMatrix(layer->GetWeightsInputGateState(), Ws);

  Matrix_t Bi(batchSize, 1); // batchSize = 2
  Bi(0,0) = 1.1; Bi(0,1) = 3.3;
  copyMatrix(layer->GetInputGateBias(), Bi);

  std::cout << "\nFor: TimeSteps = " << timeSteps << "BatchSize = " << batchSize << "StateSize = " << stateSize << "InputSize = " << inputSize << "\n";
  
  printMatrix1<Architecture>(X, "Inputs");
  printMatrix1<Architecture>(layer->GetWeightsInputGate(), "Weights");
  printMatrix1<Architecture>(layer->GetState(), "State");
  printMatrix1<Architecture>(layer->GetWeightsInputGateState(), "State weights");
  printMatrix1<Architecture>(layer->GetInputGateBias(), "Bias");

  Matrix_t di(batchSize, stateSize);
  layer->InputGate(X, di);

  printMatrix1<Architecture>(layer->GetInputGateValue(), "Input Gate Value");
  printTensor1<Architecture>(layer->GetDerivativesInput(), "Input derivative during forward pass");
  //Architecture::Rearrange(arr_XArch, XArch); // B x T x D
  //printTensor1<Architecture>(arr_XArch, "Rearrangement of XArch"); */
  //______________________________________________________________________________

  // Defining inputs.
  std::vector<TMatrixT<Double_t>> XRef(timeSteps, TMatrixT<Double_t>(batchSize, inputSize));  // T x B x D
  Tensor_t XArch, arr_XArch;

  for (size_t i = 0; i < batchSize; ++i) arr_XArch.emplace_back(timeSteps, inputSize); // B x T x D

  for (size_t i = 0; i < timeSteps; ++i) {
    randomMatrix(XRef[i]);
    XArch.emplace_back(XRef[i]);
  }

  Architecture::Rearrange(arr_XArch, XArch); // B x T x D

  Net_t lstm(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
  LSTMLayer_t* layer = lstm.AddBasicLSTMLayer(stateSize, inputSize, timeSteps);

  layer->Initialize();

  /*! unpack weights for each gate. */
  TMatrixT<Double_t> weightsInput = layer->GetWeightsInputGate();         // H x D
  randomMatrix(weightsInput, 0.0, 1.0);
  //printMatrix1<Architecture>(weightsInput, "Input weights");

  TMatrixT<Double_t> weightsCandidate = layer->GetWeightsCandidate();     // H x D
  randomMatrix(weightsCandidate, 0.0, 1.2);
  //printMatrix1<Architecture>(weightsCandidate, "Candidate weights");
  
  TMatrixT<Double_t> weightsForget = layer->GetWeightsForgetGate();       // H x D
  randomMatrix(weightsForget, 0.0, 1.4);
  //printMatrix1<Architecture>(weightsForget, "Forget weights");

  TMatrixT<Double_t> weightsOutput = layer->GetWeightsOutputGate();       // H x D
  randomMatrix(weightsForget, 0.0, 1.6);
  //printMatrix1<Architecture>(weightsOutput, "Output weights");
  
  TMatrixT<Double_t> weightsInputState = layer->GetWeightsInputGateState();       // H x H
  randomMatrix(weightsInputState, 0.0, 1.1);
  // printMatrix1<Architecture>(weightsInputState, "Input state weights");
  
  TMatrixT<Double_t> weightsCandidateState = layer->GetWeightsCandidateState();   // H x H
  randomMatrix(weightsCandidateState, 0.0, 1.3);
  // printMatrix1<Architecture>(weightsCandidateState, "Candidate state weights");
  
  TMatrixT<Double_t> weightsForgetState = layer->GetWeightsForgetGateState();     // H x H
  randomMatrix(weightsForgetState, 0.0, 1.5);
  // printMatrix1<Architecture>(weightsForgetState, "Forget state weights");
  
  TMatrixT<Double_t> weightsOutputState = layer->GetWeightsOutputGateState();     // H x H
  randomMatrix(weightsOutputState, 0.0, 1.7);
  // printMatrix1<Architecture>(weightsOutputState, "Output state weights");
  
  /*! unpack bias for each gate. */
  TMatrixT<Double_t> inputBiases = layer->GetInputGateBias();                     // H x 1
  randomMatrix(inputBiases, 0.0, 0.1);
  // printMatrix1<Architecture>(inputBiases, "Input Biases");
  
  TMatrixT<Double_t> candidateBiases = layer->GetCandidateBias();                 // H x 1
  randomMatrix(candidateBiases, 0.0, 0.3);
  // printMatrix1<Architecture>(candidateBiases, "Candidate Biases");
  
  TMatrixT<Double_t> forgetBiases = layer->GetForgetGateBias();                   // H x 1
  randomMatrix(forgetBiases, 0.0, 0.5);
  // printMatrix1<Architecture>(forgetBiases, "Forget Biases");
  
  TMatrixT<Double_t> outputBiases = layer->GetOutputGateBias();                   // H x 1
  randomMatrix(outputBiases, 0.0, 0.7);
  // printMatrix1<Architecture>(outputBiases, "Output Biases");
  
  /*! Get previous hidden state and previous cell state. */
  TMatrixT<Double_t> hiddenState = layer->GetState();       // B x H
  // printMatrix1<Architecture>(hiddenState, "Hidden State"); 
  
  TMatrixT<Double_t> cellState = layer->GetCell();          // B x H
  // printMatrix1<Architecture>(cellState, "Cell State");
  
  /*! Get each gate values. */
  TMatrixT<Double_t> inputGate = layer->GetInputGateValue();            // B x H
  randomMatrix(inputGate, 1.0, 0.3);
  //printMatrix1<Architecture>(inputGate, "Input gate values");
  
  TMatrixT<Double_t> candidateValue = layer->GetCandidateValue();       // B x H
  randomMatrix(candidateValue, 1.0, 0.5);
  // printMatrix1<Architecture>(candidateValue, "Candidate gate values");
  
  TMatrixT<Double_t> forgetGate = layer->GetForgetGateValue();          // B x H
  randomMatrix(forgetGate, 1.0, 0.7);
  // printMatrix1<Architecture>(forgetGate, "Forget gate values");
  
  TMatrixT<Double_t> outputGate = layer->GetOutputGateValue();          // B x H
  randomMatrix(outputGate, 1.0, 0.9);
  // printMatrix1<Architecture>(outputGate, "Output gate values");

  TMatrixT<Double_t> inputTmp(batchSize, stateSize);
  TMatrixT<Double_t> candidateTmp(batchSize, stateSize);
  TMatrixT<Double_t> forgetTmp(batchSize, stateSize);
  TMatrixT<Double_t> outputTmp(batchSize, stateSize);
  TMatrixT<Double_t> scale(batchSize, stateSize);
  TMatrixT<Double_t> nextHiddenState(batchSize, stateSize);

  lstm.Forward(arr_XArch);

  Tensor_t outputArch = layer->GetOutput();

  Tensor_t arr_outputArch;
  for (size_t t = 0; t < timeSteps; ++t) arr_outputArch.emplace_back(batchSize, stateSize); // T x B x H
  Architecture::Rearrange(arr_outputArch, outputArch); // B x T x H

  Double_t maximumError = 0.0;

  /*! Element-wise matrix multiplication of previous hidden
   *  state and weights of previous state followed by computing
   *  next hidden state and next cell state. */
  for (size_t t = 0; t < timeSteps; ++t) {
    inputTmp.MultT(inputGate, weightsInputState);
    inputGate.MultT(XRef[t], weightsInput);
    inputGate += inputTmp;

    candidateTmp.MultT(candidateValue, weightsCandidateState);
    candidateValue.MultT(XRef[t], weightsCandidate);
    candidateValue += candidateTmp;

    forgetTmp.MultT(forgetGate, weightsForgetState);
    forgetGate.MultT(XRef[t], weightsForget);
    forgetGate += forgetTmp;

    outputTmp.MultT(outputGate, weightsOutputState);
    outputGate.MultT(XRef[t], weightsOutput);
    outputGate += outputTmp;

    /*! Adding bias in each gate. */
    for (size_t i = 0; i < (size_t) inputGate.GetNrows(); ++i) {
      for (size_t j = 0; j < (size_t) inputGate.GetNcols(); ++j) {
        inputGate(i, j) += inputBiases(j, 0);
      }
    }
    for (size_t i = 0; i < (size_t) candidateValue.GetNrows(); ++i) {
      for (size_t j = 0; j < (size_t) candidateValue.GetNcols(); ++j) {
        candidateValue(i, j) += candidateBiases(j, 0);
      }
    }
    for (size_t i = 0; i < (size_t) forgetGate.GetNrows(); ++i) {
      for (size_t j = 0; j < (size_t) forgetGate.GetNcols(); ++j) {
        forgetGate(i, j) += forgetBiases(j, 0);
      }
    }
    for (size_t i = 0; i < (size_t) outputGate.GetNrows(); ++i) {
      for (size_t j = 0; j < (size_t) outputGate.GetNcols(); ++j) {
        outputGate(i, j) += outputBiases(j, 0);
      }
    }

    /*! Apply activation function to each computed gate values. */
    applyMatrix(inputGate, [](double i) { return sigmoid(i); });
    applyMatrix(candidateValue, [](double c) { return tanh(c); });
    applyMatrix(forgetGate, [](double f) { return sigmoid(f); });
    applyMatrix(outputGate, [](double o) { return sigmoid(o); });

    /*! Computing next cell state and next hidden state. */

    // TODO: Error has to be fixed arising due to transpose multiplication.
    // ERROR: Here it throws error:
    // Error in <MultT>: this->GetMatrixArray() == a.GetMatrixArray()
    scale.MultT(inputGate, candidateValue);
    cellState.MultT(cellState, forgetGate);
    cellState += scale;

    applyMatrix(cellState, [](double y) { return (y); });

    nextHiddenState.MultT(cellState, outputGate);
    hiddenState = nextHiddenState;

    TMatrixT<Double_t> output = arr_outputArch[t]; 

    Double_t error = maximumRelativeError(output, inputGate);
    std::cout << "Time " << t << " Error: " << error << "\n";

    maximumError = std::max(error, maximumError);
  }

  return maximumError;
}

#endif // TMVA_TEST_DNN_TEST_RNN_TEST_LSTM_FWDPASS_H
