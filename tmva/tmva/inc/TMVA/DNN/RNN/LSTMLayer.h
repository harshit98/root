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

//////////////////////////////////////////////////////////////////////////
// LSTM Network is a special kind of recurrent neural network which is  //
// capable of learning long-term dependencies. LSTM can remember given  //
// information for long period of time.                                 //
//////////////////////////////////////////////////////////////////////////

#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

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
    class TBasicLSTMLayer : public VGeneralLayer<Architecture_t>
{

public:

    using Matrix_t = typename Architecture_t::Matrix_t;
    using Scalar_t = typename Architecture_t::Scalar_t;
    using Tensor_t = std::vector<Matrix_t>;

private:

    size_t fStateSize;                      ///< Hidden state size of LSTM
    size_t fCellStateSize;                  ///< Cell state size of LSTM
    size_t fTimeSteps;                      ///< Timesteps for LSTM
    bool fRememberState;                    ///< Remember state in next pass

    DNN::EActivationFunction fF;            ///< Sigmoid activation function of the hidden state
    DNN::EActivationFunction fC;            ///< Tanh activation function of the hidden state

    Matrix_t fState;                        ///< Hidden state of LSTM
    Matrix_t fCellState;                    ///< Cell state of LSTM

    Matrix_t &fWeightsInputGate;            ///< Input weights, fWeights[0]
    Matrix_t &fWeightsInputGateState;       ///< Prev state weights, fWeights[1]
    Matrix_t &fInputGateBias;               ///< Input gate bias

    Matrix_t &fWeightsForgetGate;           ///< Input weights, fWeights[0]
    Matrix_t &fWeightsForgetGateState;      ///< Prev state weights, fWeights[1]
    Matrix_t &fForgetGateBias;              ///< Forget gate bias

    Matrix_t &fWeightsCandidate;            ///< Input weights, fWeights[0]
    Matrix_t &fWeightsCandidateState;       ///< Prev state weights, fWeights[1]
    Matrix_t &fCandidateBias;               ///< Candidate bias

    Matrix_t &fWeightsOutputGate;           ///< Input weights, fWeights[0]
    Matrix_t &fWeightsOutputGateState;      ///< Prev state weights, fWeights[1]
    Matrix_t &fOutputGateBias;              /// Output gate bias

    std::vector<Matrix_t> fDerivatives;     ///< First fDerivatives of the activations

public:

    /*! Constructor */
   TBasicLSTMLayer(size_t batchSize, size_t stateSize, size_t inputSize,
                  size_t timeSteps, bool rememberState = false,
                  DNN::EActivationFunction f = DNN::EActivationFunction::kSigmoid,
                  DNN::EActivationFunction c = DNN::EActivationFunction::kTanh,
                  bool training = true, DNN::EInitialization fA = DNN::EInitialization::kZero);

    /*! Copy Constructor */
    TBasicLSTMLayer(const TBasicLSTMLayer &);

    /*! Initialize the weights according to the given initialization
     * method. */
    // void Initialize(DNN::EInitialization m);

    /*! Initialize the state method. */
    void InitState(DNN::EInitialization m = DNN::EInitialization::kZero);

    /*! Computes and return the next hidden state 
     *  and next cell state with given input */
    void Forward(Tensor_t &input, bool isTraining = true);

    /*! Forward for a single cell (time unit) */
    void CellForward(const Matrix_t &inputGateValues, Matrix_t &forgetGateValues,
                     Matrix_t &candidateValues, Matrix_t &outputGateValues);

    /*! Backpropagates the error. Must only be called directly at the corresponding
        *  call to Forward(...). */
    void Backward(Tensor_t &gradients_backward,
                    const Tensor_t &activations_backward,
                    std::vector<Matrix_t> &inp1,
                    std::vector<Matrix_t> &inp2);

    /* Updates weights and biases, given the learning rate */
    void Update(const Scalar_t learningRate);

    /*! Backward for a single time unit
        * a the corresponding call to Forward(...). */
    inline Matrix_t & CellBackward(Matrix_t & state_gradients_backward,
                                const Matrix_t & precStateActivations,
                                const Matrix_t & input, Matrix_t & input_gradient, Matrix_t &dF);

    /*! Decides the values we'll update (NN with Sigmoid)
     * followed by candidate layer (NN with Tanh) */
    void InputGate(const Matrix_t &input, Matrix_t &dIg);

    /*! Forgets the past values (NN with Sigmoid) */
    void ForgetGate(const Matrix_t &input, Matrix_t &Fg);

    /*! Computes output values (NN with Sigmoid) */
    void OutputGate(const Matrix_t &input, Matrix_t &dOg);

    /*! Decides the new candidate values (NN with Tanh) */
    void CandidateValue(const Matrix_t &input, Matrix_t &dCv);

    /*! Prints the info about the layer */
    void Print() const;

    /*! Writes the information and the weights about the layer in an XML node. */
    virtual void AddWeightsXMLTo(void *parent);

    /*! Read the information and the weights about the layer from XML node. */
    virtual void ReadWeightsFromXML(void *parent);

    /*! Getters */
    size_t GetInputSize()               const { return this->GetInputWidth(); }
    size_t GetTimeSteps()               const { return fTimeSteps; }
    size_t GetStateSize()               const { return fStateSize; }
    size_t GetCellStateSize()           const { return fCellStateSize; }
    inline bool IsRememberState()       const { return fRememberState; }
    inline DNN::EActivationFunction     GetSigmoidActivationFunction()     const { return fF; }
    inline DNN::EActivationFunction     GetTanhActivationFunction()        const { return fC; }
    const Matrix_t                    & GetState()                         const { return fState; }
    Matrix_t                          & GetState()                               { return fState; }
    const Matrix_t                    & GetCellState()                     const { return fCellState; }
    Matrix_t                          & GetCellState()                           { return fCellState; }

    const Matrix_t                    & GetWeightsInputGate()              const { return fWeightsInputGate; }
    Matrix_t                          & GetWeightsInputGate()                    { return fWeightsInputGate; }
    const Matrix_t                    & GetWeightsCandidate()              const { return fWeightsCandidate; }
    Matrix_t                          & GetWeightsCandidate()                    { return fWeightsCandidate; }
    const Matrix_t                    & GetWeightsForgetGate()             const { return fWeightsForgetGate; }
    Matrix_t                          & GetWeightsForgetGate()                   { return fWeightsForgetGate; }
    const Matrix_t                    & GetWeightsOutputGate()             const { return fWeightsOutputGate; }
    Matrix_t                          & GetWeightsOutputGate()                   { return fWeightsOutputGate; }

    const Matrix_t                    & GetWeightsInputGateState()         const { return fWeightsInputGateState; }
    Matrix_t                          & GetWeightsInputGateState()               { return fWeightsInputGateState; }
    const Matrix_t                    & GetWeightsForgetGateState()        const { return fWeightsForgetGateState; }
    Matrix_t                          & GetWeightsForgetGateState()              { return fWeightsForgetGateState; }
    const Matrix_t                    & GetWeightsCandidateState()         const { return fWeightsCandidateState; }
    Matrix_t                          & GetWeightsCandidateState()               { return fWeightsCandidateState; }
    const Matrix_t                    & GetWeightsOutputGateState()        const { return fWeightsOutputGateState; }
    Matrix_t                          & GetWeightsOutputGateState()              { return fWeightsOutputGateState; }

    const std::vector<Matrix_t>       & GetDerivatives()                   const { return fDerivatives; }
    std::vector<Matrix_t>             & GetDerivatives()                         { return fDerivatives; }
    const Matrix_t                    & GetDerivativesAt(size_t i)         const { return fDerivatives[i]; }    
    Matrix_t                          & GetDerivativesAt(size_t i)               { return fDerivatives[i]; }

    // const Matrix_t                    & GetBiasStateGradients()            const { return fBiasGradients; }
    // Matrix_t                          & GetBiasStateGradients()                  { return fBiasGradients; }
    // const Matrix_t                    & GetWeightInputGradients()          const { return fWeightInputGradients; }
    // Matrix_t                          & GetWeightInputGradients()                { return fWeightInputGradients; }
    // const Matrix_t                    & GetWeightStateGradients()          const { return fWeightStateGradients; }
    // Matrix_t                          & GetWeightStateGradients()                { return fWeightStateGradients; }

    const Matrix_t                   & GetInputGateBiasState()         const { return fInputGateBias; }
    Matrix_t                         & GetInputGateBiasState()               { return fInputGateBias; }
    const Matrix_t                   & GetForgetGateBiasState()        const { return fForgetGateBias; }
    Matrix_t                         & GetForgetGateBiasState()              { return fForgetGateBias; }
    const Matrix_t                   & GetCandidateBiasState()         const { return fCandidateBias; }
    Matrix_t                         & GetCandidateBiasState()               { return fCandidateBias; }
    const Matrix_t                   & GetOutputGateBiasState()        const { return fOutputGateBias; }
    Matrix_t                         & GetOutputGateBiasState()              { return fOutputGateBias; }
};

//______________________________________________________________________________
//
// Basic LSTM-Layer Implementation
//______________________________________________________________________________
template <typename Architecture_t>
TBasicLSTMLayer<Architecture_t>::TBasicLSTMLayer(size_t batchSize, size_t stateSize, size_t inputSize, size_t timeSteps,
                                               bool rememberState, DNN::EActivationFunction f, DNN::EActivationFunction c,
                                               bool /* training */, DNN::EInitialization fA)
   : VGeneralLayer<Architecture_t>(batchSize, 1, timeSteps, inputSize, 1, timeSteps, stateSize, 2,
                                   {stateSize, stateSize}, {inputSize, stateSize}, 1, {stateSize}, {1}, batchSize,
                                   timeSteps, stateSize, fA),
    fTimeSteps(timeSteps),
    fStateSize(stateSize),
    fCellStateSize(stateSize),
    fRememberState(rememberState),
    fF(f),
    fC(c),
    fState(batchSize, stateSize),
    fCellState(batchSize, stateSize),

    fWeightsInputGate(this->GetWeightsAt(0)),
    fWeightsInputGateState(this->GetWeightsAt(1)),
    fInputGateBias(this->GetBiasesAt(0)),

    fWeightsForgetGate(this->GetWeightsAt(0)),
    fWeightsForgetGateState(this->GetWeightsAt(1)),
    fForgetGateBias(this->GetBiasesAt(0)),

    fWeightsCandidate(this->GetWeightsAt(0)),
    fWeightsCandidateState(this->GetWeightsAt(1)),
    fCandidateBias(this->GetBiasesAt(0)),

    fWeightsOutputGate(this->GetWeightsAt(0)),
    fWeightsOutputGateState(this->GetWeightsAt(1)),
    fOutputGateBias(this->GetBiasesAt(0))

    // fWeightInputGradients(this->GetWeightGradientsAt(0)),
    // fWeightStateGradients(this->GetWeightGradientsAt(1)),
    // fBiasGradients(this->GetBiasGradientsAt(0))
{
    for (size_t i = 0; i < timeSteps; ++i) {
        fDerivatives.emplace_back(batchSize, stateSize);
    }
    // Nothing
}

//______________________________________________________________________________
template <typename Architecture_t>
TBasicLSTMLayer<Architecture_t>::TBasicLSTMLayer(const TBasicLSTMLayer &layer)
    : VGeneralLayer<Architecture_t>(layer), 
        fTimeSteps(layer.fTimeSteps),
        fStateSize(layer.fStateSize),
        fCellStateSize(layer.fCellStateSize),
        fRememberState(layer.fRememberState),
        fF(layer.GetSigmoidActivationFunction()),
        fC(layer.GetTanhActivationFunction()),
        fState(layer.GetBatchSize(), layer.GetBatchSize()),
        fCellState(layer.GetBatchSize(), layer.GetBatchSize()),

        fWeightsInputGate(this->GetWeightsAt(0)),
        fWeightsInputGateState(this->GetWeightsAt(1)),
        fInputGateBias(this->GetBiasesAt(0)),

        fWeightsForgetGate(this->GetWeightsAt(0)),
        fWeightsForgetGateState(this->GetWeightsAt(1)),
        fForgetGateBias(this->GetBiasesAt(0)),

        fWeightsCandidate(this->GetWeightsAt(0)),
        fWeightsCandidateState(this->GetWeightsAt(1)),
        fCandidateBias(this->GetBiasesAt(0)),

        fWeightsOutputGate(this->GetWeightsAt(0)),
        fWeightsOutputGateState(this->GetWeightsAt(1)),
        fOutputGateBias(this->GetBiasesAt(0))

{
    for (size_t i = 0; i < fTimeSteps; ++i) {
        fDerivatives.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
        Architecture_t::Copy(fDerivatives[i], layer.GetDerivativesAt(i));
    }
    // Gradient matrices not copied
    Architecture_t::Copy(fState, layer.GetState());
    Architecture_t::Copy(fCellState, layer.GetCellState());
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::InputGate(const Matrix_t &input, Matrix_t &dIg)
-> void
{
    // Input gate's activation vector.
    // Input = act(W_input . input + W_state . state + bias)
    // act = Sigmoid
    const DNN::EActivationFunction fAF = this->GetSigmoidActivationFunction();
    Matrix_t tmpState(fState.GetNrows(), fState.GetNcols());
    // Matrix_t inputState(fState.GetNrows(), fState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsInputGateState);
    Architecture_t::MultiplyTranspose(fState, input, fWeightsInputGate);
    Architecture_t::ScaleAdd(fState, tmpState);
    Architecture_t::AddRowWise(fState, fInputGateBias);
    DNN::evaluateDerivative<Architecture_t>(dIg, fAF, fState);
    DNN::evaluate<Architecture_t>(fState, fAF);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::ForgetGate(const Matrix_t &input, Matrix_t &dFg)
-> void
{
    // Forget gate's activation vector.
    // Forget = act(W_input . input + W_state . state + bias)
    // act = Sigmoid
    const DNN::EActivationFunction fAF = this->GetSigmoidActivationFunction();
    Matrix_t tmpState(fState.GetNrows(), fState.GetNcols());
    // Matrix_t forgetState(fState.GetNrows(), fState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsForgetGateState);
    Architecture_t::MultiplyTranspose(fState, input, fWeightsForgetGate);
    Architecture_t::ScaleAdd(fState, tmpState);
    Architecture_t::AddRowWise(fState, fForgetGateBias);
    DNN::evaluateDerivative<Architecture_t>(dFg, fAF, fState);
    DNN::evaluate<Architecture_t>(fState, fAF);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::CandidateValue(const Matrix_t &input, Matrix_t &dCv)
-> void
{
    // C will be used to scale candidate value.
    // C = act(W_input . input + W_state . state + bias)
    // act = Tanh
    const DNN::EActivationFunction fAC = this->GetTanhActivationFunction();
    Matrix_t tmpState(fCellState.GetNrows(), fCellState.GetNcols());
    // Matrix_t candidateState(fCellState.GetNrows(), fCellState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, fCellState, fWeightsCandidateState);
    Architecture_t::MultiplyTranspose(fCellState, input, fWeightsCandidate);
    Architecture_t::ScaleAdd(fCellState, tmpState);
    Architecture_t::AddRowWise(fCellState, fCandidateBias);
    DNN::evaluateDerivative<Architecture_t>(dCv, fAC, fCellState);
    DNN::evaluate<Architecture_t>(fCellState, fAC);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::Forward(Tensor_t &input, bool /* isTraining = true */)
-> void
{
    // D : input size
    // H : state size
    // T : time size
    // B : batch size

    Tensor_t arrInput;
    for (size_t t = 0; t < fTimeSteps; ++t) arrInput.emplace_back(this->GetBatchSize(), this->GetInputWidth()); // T x B x D
    Architecture_t::Rearrange(arrInput, input);

    Tensor_t inputGateValues;
    for (size_t t = 0; t < fTimeSteps; ++t) inputGateValues.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H

    Tensor_t forgetGateValues;
    for (size_t t = 0; t < fTimeSteps; ++t) forgetGateValues.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H

    Tensor_t candidateValues;
    for (size_t t = 0; t < fTimeSteps; ++t) candidateValues.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H

    Tensor_t outputGateValues;
    for (size_t t = 0; t < fTimeSteps; ++t) outputGateValues.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H

    if (!this->fRememberState) InitState(DNN::EInitialization::kZero);

    // NOTE - The 'fState' is not same. They are different and being computed inside
    // InputGate(), ForgetGate() and OutputGate() functions.
    for (size_t t = 0; t < fTimeSteps; ++t) {
        InputGate(arrInput[t], fDerivatives[t]);
        Architecture_t::Copy(inputGateValues[t], fState);
  
        ForgetGate(arrInput[t], fDerivatives[t]);
        Architecture_t::Copy(forgetGateValues[t], fState);

        CandidateValue(arrInput[t], fDerivatives[t]);
        Architecture_t::Copy(candidateValues[t], fCellState);

        OutputGate(arrInput[t], fDerivatives[t]);
        Architecture_t::Copy(outputGateValues[t], fState);
    }

    // Retreive next cell state and next hidden state.
    Tensor_t newHiddenState;
    for (size_t t = 0; t < fTimeSteps; ++t) newHiddenState.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H

    Tensor_t newCellState;
    for (size_t t = 0; t < fTimeSteps; ++t) newCellState.emplace_back(this->GetBatchSize(), fCellStateSize); // T x B x H

    // Pass each gate values to CellForward() to calculate next
    // hidden state and next cell state.
    for(size_t t = 0; t < fTimeSteps; ++t) {
        CellForward(inputGateValues[t], forgetGateValues[t], candidateValues[t], outputGateValues[t]);
        Architecture_t::Copy(newCellState[t], fCellState);
        Architecture_t::Copy(newHiddenState[t], fState);
    }

    // Get each gate's computed values.
    Architecture_t::Rearrange(this->GetOutput(), inputGateValues); // B x T x H
    Architecture_t::Rearrange(this->GetOutput(), forgetGateValues); // B x T x H
    Architecture_t::Rearrange(this->GetOutput(), candidateValues); // B x T x H
    Architecture_t::Rearrange(this->GetOutput(), outputGateValues); // B x T x H
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::CellForward(const Matrix_t &inputGateValues, Matrix_t &forgetGateValues,
                                                        Matrix_t &candidateValues, Matrix_t &outputGateValues)
-> void
{
    // Tensor_t updateCellState;
    // for (size_t t = 0; t < fTimeSteps; ++t) updateCellState.emplace_back(this->GetBatchSize(), fCellStateSize); // T x B x H

    // Update cell state.
    Matrix_t forgetValue(fCellState.GetNrows(), fCellState.GetNcols());
    Matrix_t scaleValue(fCellState.GetNrows(), fCellState.GetNcols());
    Architecture_t::MultiplyTranspose(forgetValue, forgetGateValues, fCellState);
    Architecture_t::MultiplyTranspose(scaleValue, inputGateValues, candidateValues);
    Architecture_t::ScaleAdd(scaleValue, forgetValue);
    Architecture_t::Copy(fCellState, scaleValue);

    // Update hidden state.
    const DNN::EActivationFunction fAT = this->GetTanhActivationFunction();
    Matrix_t tmpState(fCellState.GetNrows(), fCellState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, outputGateValues, fCellState);
    Architecture_t::Copy(fState, tmpState);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::OutputGate(const Matrix_t &input, Matrix_t &dOg)
-> void
{
    // O will be used to calculate output values which will be further
    // used to calculate next hidden state.
    // O = act(W_input . input + W_state . state + bias)
    // act = Sigmoid
    const DNN::EActivationFunction fAF = this->GetSigmoidActivationFunction();
    Matrix_t tmpState(fState.GetNrows(), fState.GetNcols());
    Matrix_t outputState(fState.GetNrows(), fState.GetNcols());
    Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsOutputGateState);
    Architecture_t::MultiplyTranspose(fState, input, fWeightsOutputGate);
    Architecture_t::ScaleAdd(fState, tmpState);
    Architecture_t::AddRowWise(fState, fOutputGateBias);
    DNN::evaluateDerivative<Architecture_t>(dOg, fAF, fState);
    DNN::evaluate<Architecture_t>(fState, fAF);
}

//____________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::Backward(Tensor_t &gradients_backward,         // B x T x D
                                                    const Tensor_t &activations_backward, // B x T x D
                                                    std::vector<Matrix_t> & /*inp1*/, std::vector<Matrix_t> &
                                                    /*inp2*/) -> void
{
    // TODO
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::CellBackward(Matrix_t & state_gradients_backward,
                                                        const Matrix_t & precStateActivations,
                                                        const Matrix_t & input, Matrix_t & input_gradient, Matrix_t &dF)
-> Matrix_t &
{
    // TODO
}


//______________________________________________________________________________
template <typename Architecture_t>
auto TBasicLSTMLayer<Architecture_t>::InitState(DNN::EInitialization /*m*/) -> void
{
    DNN::initialize<Architecture_t>(this->GetState(),  DNN::EInitialization::kZero);
    DNN::initialize<Architecture_t>(this->GetCellState(),  DNN::EInitialization::kZero);
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TBasicLSTMLayer<Architecture_t>::Print() const
-> void
{
    std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
                << "Input Size: " << this->GetInputSize() << "\n"
                << "Hidden State Size: " << this->GetStateSize() << "\n"
                << "Cell State Size: " << this->GetCellStateSize() << "\n";
}

// template <typename Architecture_t>
// auto debugMatrix(const typename Architecture_t::Matrix_t &A, const std::string name = "matrix")
// -> void
// {
    // std::cout << name << "\n";
    // for (size_t i = 0; i < A.GetNrows(); ++i) {
        // for (size_t j = 0; j < A.GetNcols(); ++j) {
            // std::cout << A(i, j) << " ";
        // }
        // std::cout << "\n";
    // }
    // std::cout << "********\n";
// }

//______________________________________________________________________________
template <typename Architecture_t>
void TBasicLSTMLayer<Architecture_t>::AddWeightsXMLTo(void *parent)
{
    auto layerxml = gTools().xmlengine().NewChild(parent, 0, "LSTMLayer");

    // Write all other info like stateSize, cellSize, inputSize, timeSteps, rememberState
    gTools().xmlengine().NewAttr(layerxml, 0, "StateSize", gTools().StringFromInt(this->GetStateSize()));
    gTools().xmlengine().NewAttr(layerxml, 0, "CellSize", gTools().StringFromInt(this->GetCellStateSize()));
    gTools().xmlengine().NewAttr(layerxml, 0, "InputSize", gTools().StringFromInt(this->GetInputSize()));
    gTools().xmlengine().NewAttr(layerxml, 0, "TimeSteps", gTools().StringFromInt(this->GetTimeSteps()));
    gTools().xmlengine().NewAttr(layerxml, 0, "RememberState", gTools().StringFromInt(this->IsRememberState()));

    // Write weights and bias matrices
    this->WriteMatrixToXML(layerxml, "InputWeights", this->GetWeightsAt(0));
    this->WriteMatrixToXML(layerxml, "StateWeights", this->GetWeightsAt(1));
    this->WriteMatrixToXML(layerxml, "Biases", this->GetBiasesAt(0));
}

//______________________________________________________________________________
template <typename Architecture_t>
void TBasicLSTMLayer<Architecture_t>::ReadWeightsFromXML(void *parent)
{
    // Read weights and biases
    this->ReadMatrixXML(parent,"InputWeights", this->GetWeightsAt(0));
    this->ReadMatrixXML(parent,"StateWeights", this->GetWeightsAt(1));
    this->ReadMatrixXML(parent,"Biases", this->GetBiasesAt(0));
}

} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif // LSTM_LAYER_H
