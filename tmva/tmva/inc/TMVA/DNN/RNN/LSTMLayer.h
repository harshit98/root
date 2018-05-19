// @(#)root/tmva/tmva/dnn/rnn:$Id$
// Author: Harshit Prasad 17/05/18

/**********************************************************************************
 * Project: TMVA - a ROOT-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class : TBasicLSTMLayer                                                        *
 *                                                                                *
 * Description:                                                                   *
 *       Long Short Term Memory (LSTM) Layer                                      *
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

///////////////////////////////////////////////////////////////////////
// LSTM Network is a special kind of recurrent neural network which is
// capable of learning long-term dependencies. LSTM can remember given
// information for long period of time.
///////////////////////////////////////////////////////////////////////

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

    Matrix_t fMemoryState; // for updating memory
    Matrix_t fHiddenState; // for updating hidden state

    Matrix_t fInputGateState; // Hidden state of Input Gate
    Matrix_t fCandidateGateState; // Hidden state of Candidate Gate
    Matrix_t fForgetGateState; // Hidden state of Forget Gate
    Matrix_t fOutputGateState; // Hidden state of Output Gate

    Matrix_t &fInputWeightsOfInputGate; // Input gate weights, fInputWeightsOfInputGate[0]
    Matrix_t &fWeightsInputState; // Previous state weights, fWeightsInputState[1]
    Matrix_t &fInputGateBiases; // Input gate biases

    Matrix_t &fInputWeightsOfCandidate; // Input candidate weights, fInputWeightsOfCandidate[0]
    Matrix_t &fWeightsCandidateState; // Previous state weights, fWeightsCandidateState[1]
    Matrix_t &fCandidateBiases; // Candidate biases

    Matrix_t &fInputWeightsOfForgetGate; // Forget gate weights, fInputWeightsOfForgetGate[0]
    Matrix_t &fWeightsForgetState; // Previous state weights, fWeightsForgetState[1]
    Matrix_t &fForgetGateBiases; // Forget gate biases 

    Matrix_t &fInputWeightsOfOutputGate; // Output gate weights, fInputWeightsOfOutputGate[0]
    Matrix_t &fWeightsOutputState; // Output gate weights, fWeightsOutputState[1]
    Matrix_t &fOutputGateBiases; // Output gate biases

    // derivatives input gate, forget gate and output gate
    // TODO
    // .....
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

    /* Updates Memory cell value  */
    void UpdateMemoryCell(const Matrix_t &input, Matrix_t &dF, Matrix_t &A);

    /* Updates next hidden state */
    void UpdateHiddenState(const Matrix_t &C_T, Matrix_t &A);

    /* Computes candidate values (NN with Tanh) */
    void CandidateLayer(const Matrix_t &input, Matrix_t &dF);

    /* Computes and return the next state with given input */
    void Forward(Tensor_t &input, bool isTraining = true);

   /*! Backpropagates the error. Must only be called directly at the corresponding
    *  call to Forward(...). */
   void Backward(Tensor_t &gradients_backward,
                 const Tensor_t &activations_backward,
                 std::vector<Matrix_t> &inp1,
                 std::vector<Matrix_t> &inp2);

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
/* template<typename Architecture_t>
TBasicLSTMLayer<Architecture_t>::TBasicLSTMLayer(size_t batchSize, size_t stateSize, size_t inputSize,
                                              size_t timeSteps, bool rememberState,
                                              DNN::EActivationFunction f,
                                              bool training, DNN::EInitialization fA)
    : VGeneralLayer<Architecture_t>(batchSize, 1, 1, inputSize, 1, 1, stateSize, 2, 
                                    {stateSize, stateSize}, {inputSize, stateSize}, 1,
                                    {stateSize}, {1}, batchSize, timeSteps, stateSize, fA)
     fTimeSteps(timeSteps),
     fStateSize(stateSize),
     fRememberState(rememberState),
     fF(f),
     fState(batchSize, stateSize),
     fWeightsInput(this->GetWeightsAt(0)),
     fWeightsState(this->GetWeightsAt(1)),
     fBiases(this->GetBiasesAt(0)),
     fWeightInputGradients(this->GetWeightGradientsAt(0)),
     fWeightStateGradients(this->GetWeightGradientsAt(1)),
     fBiasGradients(this->GetBiasGradientsAt(0))
{
  for (size_t i = 0; i < timeSteps; ++i) {
     fDerivatives.emplace_back(batchSize, stateSize);
  }
   // Nothing
}

//______________________________________________________________________________
template <typename Architecture_t>
TBasicLSTMLayer<Architecture_t>::TBasicLSTMLayer(const TBasicLSTMLayer &layer)
   : VGeneralLayer<Architecture_t>(layer), fTimeSteps(layer.fTimeSteps), fStateSize(layer.fStateSize),
     fRememberState(layer.fRememberState), fF(layer.GetActivationFunction()),
     fState(layer.GetBatchSize(), layer.GetStateSize()), fWeightsInput(this->GetWeightsAt(0)),
     fWeightsState(this->GetWeightsAt(1)), fBiases(this->GetBiasesAt(0)),
     fDerivatives(), fWeightInputGradients(this->GetWeightGradientsAt(0)),
     fWeightStateGradients(this->GetWeightGradientsAt(1)), fBiasGradients(this->GetBiasGradientsAt(0))
{
   for (size_t i = 0; i < fTimeSteps; ++i) {
     fDerivatives.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
     Architecture_t::Copy(fDerivatives[i], layer.GetDerivativesAt(i));
   }
   // Gradient matrices not copied
   Architecture_t::Copy(fState, layer.GetState());
} */

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::InputGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // I is input gate's activation vector
    // I = act(W_input . input + W_state . state + bias)
    // act = Sigmoid
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fInputGateState.GetNrows(), fInputGateState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, fInputGateState, fWeightsInputState);
    Architecture_t::MultiplyTranspose(fInputGateState, input, fInputWeightsOfInputGate);
    Architecture_t::ScaleAdd(fInputGateState, tmpState);
    Architecture_t::AddRowWise(fInputGateState, fInputGateBiases);
    DNN::evaluateDerivative<Architecture_t>(dF, fAF, fInputGateState);
    DNN::evaluate<Architecture_t>(fInputGateState, fAF);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::CandidateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // C is candidate values
    // C = act(W_input . input + W_state . prev_state + bias)
    // act = Tanh
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fCandidateGateState.GetNrows(), fCandidateGateState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, fCandidateGateState, fWeightsCandidateState);
    Architecture_t::MultiplyTranspose(fCandidateGateState, input, fInputWeightsOfCandidate);
    Architecture_t::ScaleAdd(fCandidateGateState, tmpState);
    Architecture_t::AddRowWise(fCandidateGateState, fCandidateBiases);
    DNN::evaluateDerivative<Architecture_t>(dF, fAF, fCandidateGateState);
    DNN::evaluate<Architecture_t>(fCandidateGateState, fAF);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::UpdateMemoryCell(const Matrix_t &input, Matrix_t &dF, Matrix_t &A)
-> void
{
    /*! Memory cell value C_t will be calculated using candidate state values,
     *  input gate values and forget gate values. C_t will be passed to next timestep. */
    Matrix_t C = CandidateGateLayer(input, dF);
    Matrix_t I = InputGateLayer(input, dF);
    Matrix_t F = ForgetGateLayer(input, dF);
    Matrix_t C_T;
    Matrix_t tmpState(fMemoryState.GetNrows(), fMemorysState.GetNcols());

    for(size_t i = 1; i <= fTimeSteps; ++i) {
        for(size_t t = 1; t <= fTimeSteps; ++t) {
            Architecture_t::MultiplyTranspose(tmpState, F[t], C_T[t-1]);
            Architecture_t::MultiplyTranspose(F[t], I[t], C[t]);
            C_T[t-1] = Architecture_t::ScaleAdd(tmpState, F[t]);
        }
        C_T[i] += C_T[i-1];
    }
    UpdateHiddenState(C_T, A); // A is O_t matrix from OutputLayerGate()
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::UpdateHiddenState(const Matrix_t &C_T, Matrix_t &A)
-> void
{
    /*! Next hidden state values h_t will be calculated using memory state values C_t
     *  and output gate values. h_t will be passed to next timestep. */
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fHiddenState.GetNrows(), fHiddenState.GetNcols()); // tmpState = h_t
    Architecture_t::MultiplyTranspose(tmpState, A, C_T); 
}
//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::ForgetGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // F is forget gate's activation vector
    // F = act(W_input . input + W_state . state + bias)
    // act = Sigmoid
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fForgetGateState.GetNrows(), fForgetGateState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, fForgetGateState, fWeightsForgetState);
    Architecture_t::MultiplyTranspose(fForgetGateState, input, fInputWeightsOfForgetGate);
    Architecture_t::ScaleAdd(fForgetGateState, tmpState);
    Architecture_t::AddRowWise(fForgetGateState, fForgetGateBiases);
    DNN::evaluateDerivative<Architecture_t>(dF, fAF, fForgetGateState);
    DNN::evaluate<Architecture_t>(fForgetGateState, fAF);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::OutputGateLayer(const Matrix_t &input, Matrix_t &dF)
-> void
{
    // out is output gate's activation vector
    // out = act(W_input . input + W_state . state + bias)
    // act = Sigmoid
    const DNN::EActivationFunction fAF = this.GetActivationFunction();
    Matrix_t tmpState(fOutputGateState.GetNrows(), fOutputGateState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, fOutputGateState, fWeightsOutputState);
    Architecture_t::MultiplyTranspose(fOutputGateState, input, fInputWeightsOfOutputGate);
    Architecture_t::ScaleAdd(fOutputGateState, tmpState);
    Architecture_t::AddRowWise(fOutputGateState, fOutputGateBiases);
    DNN::evaluateDerivative<Architecture_t>(dF, fAF, fOutputGateState);
    DNN::evaluate<Architecture_t>(fOutputGateState, fAF);
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
      // InputGateLayer(arrInput[t], fDerivatives[t]);
      // CandidateLayer(arrInput[t], fDerivatives[t]);
      // ForgetGateLayer(arrInput[t], fDerivatives[t]);
      OutputGateLayer(arrInput[t], fDerivatives[t]);
      UpdateMemoryCell(arrInput[t], fDerivatives[t]);
      Architecture_t::Copy(arrOutput[t], fState);
   }
   Architecture_t::Rearrange(this->GetOutput(), arrOutput);  // B x T x D
}

//____________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::Backward(Tensor_t &gradients_backward,          // B x T x D
                                                     const Tensor_t &activations_backward,  // B x T x D 
                                                     std::vector<Matrix_t> &inp1,
                                                     std::vector<Matrix_t> &inp2)   
-> void
{
   //TODO
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TBasicRNNLayer<Architecture_t>::InitState(DNN::EInitialization /*m*/) -> void
{
   DNN::initialize<Architecture_t>(this->GetState(),  DNN::EInitialization::kZero);
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
