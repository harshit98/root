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

    // initialize variables here
    Matrix_t fState; // Hidden state
    Matrix_t &fWeightsInput; // Input weights, fWeights[0]
    Matrix_t &fWeightsState; // Previous state weights, fWeights[1]
    Matrix_t &fBiases; // Biases

    // .....

public:

    /* Constructor */
    TBasicLSTMLayer(size_t batchSize, size_t stateSize, size_t inputSize,
              size_t timeSteps, bool rememberState = false,
              DNN::EActivationFunction f = DNN::EActivationFunction::kTanh,
              bool training = true, DNN::EInitialization fA = DNN::EInitialization::kZero);

    /* Copy Constructor */
    TBasicLSTMLayer(const TBasicLSTMLayer &);

    /* Destructor */
    ~TBasicLTSMLayer();

    /* Updates weights and biases, according to learning rate  */
    void Update(const Scalar_t learningRate);

    /* Prints the info about the layer */
    void Print() const;

    /* Writes the information and weights about the layer in an XML node */
    virtual void AddWeightsXMLTo(void *parent);

    /* Reads the information and weights about the layer from an XML node */
    virtual void ReadWeightsFromXML(void *parent);

    /* Getters */

};

//______________________________________________________________________________
//
// BasicLSTMLayer Implementation
//______________________________________________________________________________
template <typename Architecture_t>
// auto inline TBasicLSTMLayer<Architecture_t>::TBasicLSTMLayer

//______________________________________________________________________________
template <typename Architecture_t>
// TODO - Candidate Value can_t
auto inline TBasicLSTMLayer<Architecture_t>::InputGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // i_t is input gate's activation vector
    // i_t = act(W_input . input + W_state . prev_state + bias)
    // act = sigmoid
    // calculate candidate state value here
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fState.GetNRows(), fState.GetNCols());
    Architecture_t::MultiplyTranspose(/* tmpState, fState, fWeightsState */);
    Architecture_t::MultiplyTranspose(/* fState, input, fWeightsInput */);
    Architecture_t::ScaleAdd(/* fState, tmpState */);
    Architecture_t::AddRowWise(/* fState, fBiases */);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::MemoryCell(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // Memory cell value C_t has been calculated using candiate state value,
    // input gate value and the forget gate value
    // TODO
    // Write the algorithm to calculate memory cell value.
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::ForgetGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // f_t is forget gate's activation vector
    // f_t = act(W_input . input + W_state . prev_state + bias)
    // act = sigmoid
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fState.GetNRows(), fState.GetNCols());
    Architecture_t::MultiplyTranspose(/* tmpState, fState, fWeightsState */);
    Architecture_t::MultiplyTranspose(/* fState, input, fWeightsInput */);
    Architecture_t::ScaleAdd(/* fState, tmpState */);
    Architecture_t::AddRowWise(/* fState, fBiases */);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::OutputGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // TODO
    // To calculate the value of output gate, we should be having memory cell value
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
