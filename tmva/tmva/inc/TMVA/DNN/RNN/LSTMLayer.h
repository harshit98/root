// @(#)root/tmva/tmva/dnn/rnn:$Id$
// Author: Harshit Prasad 17/05/18

/**********************************************************************************
 * Project: TMVA - a ROOT-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class : TBasicLSTMLayer                                                         *
 *                                                                                *
 * Description:                                                                   *
 *       Long Short Term Memory (LSTM) Layer                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *       Harshit Prasad    <harshitprasad28@gmail.com> - CERN, Switzerland        *
 *                                                                                *
 * Copyright (c) 2005-2018:                                                       *
 * All rights reserved.                                                           *
 *       CERN, Switzerland                                                        *
 *                                                                                *
 * For the licensing terms see $ROOTSYS/LICENSE.                                  *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                      *
 **********************************************************************************/

//////////////////////////////////////////////////////////////////////
// LSTM Network is a special kind of Recurrent Neural Network (RNN) 
// which capable of learning long-term dependencies. LSTM can remember
// given information for long period of time.
//////////////////////////////////////////////////////////////////////

#ifndef LSTMLAYER_H
#define LSTMLAYER_H

#include <cmath>
#include <iostream>
#include <vector>

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"

namespace TMVA {
namespace DNN {
namespace RNN {
//______________________________________________________________________________
//
// Basic LSTM Layer
//______________________________________________________________________________

/** \class BasicLSTMLayer
        
    Generic LSTM Layer class

    This generic LSTM Layer class represents a special RNN
    layer. It inherits properties of the generic virtual base
    class VGeneralLayer.

*/
template <typename Architecture_t>
class TBasicLSTMLayer : public VGeneralLayer<Architecture_t> {

public:

    using Matrix_t = typename Architecture_t::Matrix_t;
    using Scalar_t = typename Architecture_t::Scalar_t;
    using Tensor_t = std::vector<Matrix_t>;

private:

    size_t fTimeSteps; // Timesteps for LSTM
    size_t fRememberState; // Remember state in next pass
    size_t fStateSize; // Hidden state of LSTM

    DNN::EActivationFunction fF; // Activation function of hidden state

    Matrix_t fState; // Hidden state
    Matrix_t &fWeightsInput; // Input weights, fWeights[0]
    Matrix_t &fWeightsState; // Previous state weights, fWeights[1]
    Matrix_t &fBiases; // Biases

   std::vector<Matrix_t> fDerivatives; ///< First fDerivatives of the activations
   Matrix_t &fWeightInputGradients; ///< Gradients w.r.t. the input weights
   Matrix_t &fWeightStateGradients; ///< Gradients w.r.t. the state weights
   Matrix_t &fBiasGradients;        ///< Gradients w.r.t. the bias values

public:

    /* Constructor */
    TBasicLSTMLayer(size_t batchSize, size_t stateSize, size_t inputSize,
                    size_t timeSteps, bool rememberState = false,
                    DNN::EActivationFunction f = DNN::EActivationFunction::kTanh,
                    bool training = true, DNN::EInitialization fA = DNN::EInitialization::kZero);

    /* Copy Constructor */
    TBasicLSTMLayer(const TBasicLSTMLayer &);

    /*  Initialize the weights according to the given initialization
    **  method. */
    // void Initialize(DNN::EInitialization m);

    /* Initialize the state method. */
    void InitState(DNN::EInitialization m = DNN::EInitialization::kZero);

    /*! Decides the values we'll update (NN with Sigmoid)  
    *  followed by Candidate Layer (NN with Tanh) */
    void InputGateLayer(const Matrix_t &input, Matrix_t & dF);

    /*! Forgets long term dependencies or reset the memory 
    *  It is NN with Sigmoid */ 
    void ForgetGateLayer(const Matrix_t &input, Matrix_t &dF);

    /* Computes output values */
    void OutputGateLayer(const Matrix_t &input, Matrix_t &dF);

    /* Updates Memory cell value */
    void UpdateMemoryCell(const Matrix_t &input, Matrix_t &dF);

    /* Computes candidate values (NN with Tanh)*/
    void CandidateLayer(const Matrix_t &input, Matrix_t &dF);

    /* Computes and return the next state with given input */
    void Forward(Tensor_t &input, bool isTraining = true); 

    /* Updates weights and biases, according to learning rate  */
    void Update(const Scalar_t learningRate);

    /* Prints the info about the layer */
    void Print() const;

    /* Writes the information and weights about the layer in an XML node */
    virtual void AddWeightsXMLTo(void *parent);

    /* Reads the information and weights about the layer from an XML node */
    virtual void ReadWeightsFromXML(void *parent);

   /** Getters */
   size_t GetTimeSteps() const { return fTimeSteps; }
   size_t GetStateSize() const { return fStateSize; }
   size_t GetInputSize() const { return this->GetInputWidth(); }
   inline bool IsRememberState()  const {return fRememberState;}
   inline DNN::EActivationFunction GetActivationFunction()  const {return fF;}
   Matrix_t        & GetState()            {return fState;}
   const Matrix_t & GetState()       const  {return fState;}
   Matrix_t        & GetWeightsInput()        {return fWeightsInput;}
   const Matrix_t & GetWeightsInput()   const {return fWeightsInput;}
   Matrix_t        & GetWeightsState()        {return fWeightsState;}
   const Matrix_t & GetWeightsState()   const {return fWeightsState;}
   std::vector<Matrix_t>       & GetDerivatives()        {return fDerivatives;}
   const std::vector<Matrix_t> & GetDerivatives()   const {return fDerivatives;}
   Matrix_t &GetDerivativesAt(size_t i) { return fDerivatives[i]; }
   const Matrix_t &GetDerivativesAt(size_t i) const { return fDerivatives[i]; }
   Matrix_t        & GetBiasesState()              {return fBiases;}
   const Matrix_t & GetBiasesState()         const {return fBiases;}
   Matrix_t        & GetBiasStateGradients()            {return fBiasGradients;}
   const Matrix_t & GetBiasStateGradients() const {return fBiasGradients;}
   Matrix_t        & GetWeightInputGradients()         {return fWeightInputGradients;}
   const Matrix_t & GetWeightInputGradients()    const {return fWeightInputGradients;}
   Matrix_t        & GetWeightStateGradients()         {return fWeightStateGradients;}
   const Matrix_t & GetWeightStateGradients()    const {return fWeightStateGradients;}
};

//______________________________________________________________________________
//
// BasicLSTMLayer Implementation
//______________________________________________________________________________
template <typename Architecture_t>
// auto inline TBasicLSTMLayer<Architecture_t>::TBasicLSTMLayer

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::InputGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // I is input gate's activation vector
    // I = act(W_input . input + W_state . prev_state + bias)
    // act = Sigmoid
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fState.GetNRows(), fState.GetNCols());
    Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsState);
    Architecture_t::MultiplyTranspose(fState, input, fWeightsInput);
    Architecture_t::ScaleAdd( fState, tmpState);
    Architecture_t::AddRowWise(fState, fBiases);
    DNN::evaluateDerivative<Architecture_t>(dF, fAF, fState);
    DNN::evaluate<Architecture_t>(fState, fAF);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTM<Architecture_t>::CandidateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // C is candidate values
    // C = act(W_input . input + W_state . prev_state + bias)
    // act = Tanh
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fState.GetNRows(), fState.GetNCols());
    Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsState);
    Architecture_t::MultiplyTranspose(fState, input, fWeightsInput);
    Architecture_t::ScaleAdd( fState, tmpState);
    Architecture_t::AddRowWise(fState, fBiases);
    DNN::evaluateDerivative<Architecture_t>(dF, fAF, fState);
    DNN::evaluate<Architecture_t>(fState, fAF);

    // pass candidate values in UpdateMemoryCell() to update memory
    // TODO
    // ....
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::UpdateMemoryCell(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // Memory cell value M has been calculated using candidate state value,
    // TODO
    // Write the algorithm to calculate memory cell value.
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::ForgetGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // F is forget gate's activation vector
    // F = act(W_input . input + W_state . prev_state + bias)
    // act = Sigmoid
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fState.GetNRows(), fState.GetNCols());
    Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsState);
    Architecture_t::MultiplyTranspose(fState, input, fWeightsInput);
    Architecture_t::ScaleAdd( fState, tmpState);
    Architecture_t::AddRowWise(fState, fBiases);
    DNN::evaluateDerivative<Architecture_t>(dF, fAF, fState);
    DNN::evaluate<Architecture_t>(fState, fAF);
    // pass the values in UpdateMemoryCell() to update memory
    // TODO
    // ....
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::OutputGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // TODO
    // To calculate the value of output gate, we should be having memory cell value
    // This function will calculate output value and next hidden state.
    // .....
}

template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::Forward(Tensor_t &input, bool /* isTraining */)
-> void
{
   // D : input size
   // H : state size
   // T : time size
   // B : batch size
   
   Tensor_t arrInput;
   for (size_t t = 0; t < fTimeSteps; ++t) arrInput.emplace_back(this->GetBatchSize(), this->GetInputWidth()); // T x B x D
   Architecture_t::Rearrange(arrInput, input);
   Tensor_t arrOutput;
   for (size_t t = 0; t < fTimeSteps;++t) arrOutput.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H 

   if (!this->fRememberState) InitState(DNN::EInitialization::kZero);
   for (size_t t = 0; t < fTimeSteps; ++t) {
      InputGateLayer(arrInput[t], fDerivatives[t]);
      CandidateLayer(arrInput[t], fDerivatives[t]);
      ForgetGateLayer(arrInput[t], fDerivatives[t]);
      Architecture_t::Copy(arrOutput[t], fState);
   }
   Architecture_t::Rearrange(this->GetOutput(), arrOutput);  // B x T x D
}

//______________________________________________________________________________
template <typename Architecture_t>
auto debugMatrix(const typename Architecture_t::Matrix_t &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t i = 0; i < A.GetNrows(); ++i) {
    for (size_t j = 0; j < A.GetNcols(); ++j) {
        std::cout << A(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "********\n";
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TBasicLSTMLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
             << "Input Size: " << this->GetInputSize() << "\n"
             << "Hidden State Size: " << this->GetStateSize() << "\n";
}

//______________________________________________________________________________
template <typename Architecture_t>
void TBasicLSTMLayer<Architecture_t>::AddWeightsXMLTo(void *parent)
{
   auto layerxml = gTools().xmlengine().NewChild(parent, 0, "LSTMLayer");

   // Write LSTM Layer information: stateSize, inputSize, timeSteps, rememberState
   gTools().xmlengine().NewAttr(layerxml, 0, "StateSize", gTools().StringFromInt(this->GetStateSize()));
   gTools().xmlengine().NewAttr(layerxml, 0, "InputSize", gTools().StringFromInt(this->GetInputSize()));
   gTools().xmlengine().NewAttr(layerxml, 0, "TimeSteps", gTools().StringFromInt(this->GetTimeSteps()));
   gTools().xmlengine().NewAttr(layerxml, 0, "RememberState", gTools().StringFromInt(this->IsRememberState()));

   // Write weights and biases matrices
   this->WriteMatrixToXML(layerxml, "InputWeights", this -> GetWeightsAt(0));
   this->WriteMatrixToXML(layerxml, "StateWeights", this -> GetWeightsAt(1));
   this->WriteMatrixToXML(layerxml, "Biases",  this -> GetBiasesAt(0));
}

//______________________________________________________________________________
template <typename Architecture_t>
void TBasicLSTMLayer<Architecture_t>::ReadWeightsFromXML(void *parent)
{
   // Read weights and biases
   this->ReadMatrixXML(parent,"InputWeights", this -> GetWeightsAt(0));
   this->ReadMatrixXML(parent,"StateWeights", this -> GetWeightsAt(1));
   this->ReadMatrixXML(parent,"Biases", this -> GetBiasesAt(0));
}

} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif
