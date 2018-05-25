// @(#)root/tmva/tmva/dnn:$Id$ 
// Author: Saurav Shekhar 23/06/17

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar, Harshit Prasad                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Implementation of the functions required for the forward and    //
// backward propagation of activations through a recurrent neural  //
// network in the reference implementation.                        //
/////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA {
namespace DNN  {

  
//______________________________________________________________________________
template<typename Scalar_t>
auto TReference<Scalar_t>::RecurrentLayerBackward(TMatrixT<Scalar_t> & state_gradients_backward, // BxH
                                                  TMatrixT<Scalar_t> & input_weight_gradients,
                                                  TMatrixT<Scalar_t> & state_weight_gradients,
                                                  TMatrixT<Scalar_t> & bias_gradients,
                                                  TMatrixT<Scalar_t> & df, //BxH
                                                  const TMatrixT<Scalar_t> & state, // BxH
                                                  const TMatrixT<Scalar_t> & weights_input, // HxD 
                                                  const TMatrixT<Scalar_t> & weights_state, // HxH
                                                  const TMatrixT<Scalar_t> & input,  // BxD
                                                  TMatrixT<Scalar_t> & input_gradient)
-> Matrix_t &
{

   // std::cout << "Reference Recurrent Propo" << std::endl;
   // std::cout << "df\n";
   // df.Print();
   // std::cout << "state gradient\n";
   // state_gradients_backward.Print();
   // std::cout << "inputw gradient\n";
   // input_weight_gradients.Print(); 
   // std::cout << "state\n";
   // state.Print();
   // std::cout << "input\n";
   // input.Print();
   
   // Compute element-wise product.
   for (size_t i = 0; i < (size_t) df.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t) df.GetNcols(); j++) {
         df(i,j) *= state_gradients_backward(i,j);      // B x H
      }
   }
   
   // Input gradients.
   if (input_gradient.GetNoElements() > 0) {
      input_gradient.Mult(df, weights_input);     // B x H . H x D = B x D
   }
   // State gradients
   if (state_gradients_backward.GetNoElements() > 0) {
      state_gradients_backward.Mult(df, weights_state);  // B x H . H x H = B x H
   }
   
   // Weights gradients.
   if (input_weight_gradients.GetNoElements() > 0) {
      TMatrixT<Scalar_t> tmp(input_weight_gradients);
      input_weight_gradients.TMult(df, input);             // H x B . B x D
      input_weight_gradients += tmp;
   }
   if (state_weight_gradients.GetNoElements() > 0) {
      TMatrixT<Scalar_t> tmp(state_weight_gradients);
      state_weight_gradients.TMult(df, state);             // H x B . B x H
      state_weight_gradients += tmp;
   }
   
   // Bias gradients. B x H -> H x 1
   if (bias_gradients.GetNoElements() > 0) {
      // this loops on state size
      for (size_t j = 0; j < (size_t) df.GetNcols(); j++) {
         Scalar_t sum = 0.0;
         // this loops on batch size summing all gradient contributions in a batch
         for (size_t i = 0; i < (size_t) df.GetNrows(); i++) {
            sum += df(i,j);
         }
         bias_gradients(j,0) += sum;
      }
   }

   // std::cout << "RecurrentPropo: end " << std::endl;

   // std::cout << "state gradient\n";
   // state_gradients_backward.Print();
   // std::cout << "inputw gradient\n";
   // input_weight_gradients.Print(); 
   // std::cout << "bias gradient\n";
   // bias_gradients.Print(); 
   // std::cout << "input gradient\n";
   // input_gradient.Print(); 

   
   return input_gradient;
}

//______________________________________________________________________________
template <typename Scalar_t>
auto TReference<Scalar_t>::LSTMLayerBackward(const TMatrixT<Scalar_t> & /* state_gradients_backward*/,
                                             TMatrixT<Scalar_t> & input_weight_gradients,
                                             TMatrixT<Scalar_t> & forget_weight_gradients,
                                             TMatrixT<Scalar_t> & candidate_weight_gradients,
                                             TMatrixT<Scalar_t> & output_weight_gradients,
                                             TMatrixT<Scalar_t> & input_state_weight_gradients,
                                             TMatrixT<Scalar_t> & forget_state_weight_gradients,
                                             TMatrixT<Scalar_t> & candidate_state_weight_gradients,
                                             TMatrixT<Scalar_t> & output_state_weight_gradients,
                                             TMatrixT<Scalar_t> & input_bias_gradients,
                                             TMatrixT<Scalar_t> & forget_bias_gradients,
                                             TMatrixT<Scalar_t> & candidate_bias_gradients,
                                             TMatrixT<Scalar_t> & output_bias_gradients,
                                             TMatrixT<Scalar_t> & dIg,
                                             TMatrixT<Scalar_t> & dCv,
                                             TMatrixT<Scalar_t> & dFg,
                                             TMatrixT<Scalar_t> & dOg,
                                             const TMatrixT<Scalar_t> & output_state,
                                             const TMatrixT<Scalar_t> & /* cell_state */,
                                             const TMatrixT<Scalar_t> & weights_input,
                                             const TMatrixT<Scalar_t> & weights_forget,
                                             const TMatrixT<Scalar_t> & weights_candidate,
                                             const TMatrixT<Scalar_t> & weights_output,
                                             const TMatrixT<Scalar_t> & /* weights_input_state */,
                                             const TMatrixT<Scalar_t> & /* weights_forget_state */,
                                             const TMatrixT<Scalar_t> & /* weights_candidate_state */,
                                             const TMatrixT<Scalar_t> & /* weights_output_state */,
                                             const TMatrixT<Scalar_t> & input,
                                             TMatrixT<Scalar_t> & input_gradient,
                                             TMatrixT<Scalar_t> & forget_gradient,
                                             TMatrixT<Scalar_t> & candidate_gradient,
                                             TMatrixT<Scalar_t> & output_gradient)
-> Matrix_t & 
{
    /* TODO: Update all gate values during backward pass using required equations.
    * Reference: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9 */
    
    // Input gradients
    if (input_gradient.GetNoElements() > 0) {
        input_gradient.Mult(dIg, weights_input);
    }
    if (forget_gradient.GetNoElements() > 0) {
        forget_gradient.Mult(dFg, weights_forget);
    }
    if (candidate_gradient.GetNoElements() > 0) {
        candidate_gradient.Mult(dCv, weights_candidate);
    }
    if (output_gradient.GetNoElements() > 0) {
        output_gradient.Mult(dOg, weights_output);
    }

    // State gradients
    // if (input_state_gradients_backward.GetNoElements() > 0) {
        // input_state_gradients_backward.Mult(dIg, weights_input_state);
    // }
    // if (forget_state_gradients_backward.GetNoElements() > 0) {
        // forget_state_gradients_backward.Mult(dFg, weights_forget_state);
    // }
    // if (candidate_state_gradients_backward.GetNoElements() > 0) {
        // candidate_state_gradients_backward.Mult(dCv, weights_candidate_state);
    // }
    // if (output_state_gradients_backward.GetNoElements() > 0) {
        // output_state_gradients_backward.Mult(dOg, weights_output_state);
    // }

    // Weight gradients
    // Total there are 8 different weight matrices.

    // For input gate.
    if (input_weight_gradients.GetNoElements() > 0) {
        TMatrixT<Scalar_t> tmp(input_weight_gradients);
        input_weight_gradients.TMult(dIg, input);
        input_weight_gradients += tmp;
    }
    if (input_state_weight_gradients.GetNoElements() > 0) {
        TMatrixT<Scalar_t> tmp(input_state_weight_gradients);
        input_state_weight_gradients.TMult(dIg, output_state);
        input_state_weight_gradients += tmp;
    }

    // For forget gate.
    if (forget_weight_gradients.GetNoElements() > 0) {
        TMatrixT<Scalar_t> tmp(forget_weight_gradients);
        forget_weight_gradients.TMult(dFg, input);
        forget_weight_gradients += tmp;
    }
    if (forget_state_weight_gradients.GetNoElements() > 0) {
        TMatrixT<Scalar_t> tmp(forget_state_weight_gradients);
        forget_state_weight_gradients.TMult(dFg, output_state);
        forget_state_weight_gradients += tmp;
    }

    // For candidate gate.
    if (candidate_weight_gradients.GetNoElements() > 0) {
        TMatrixT<Scalar_t> tmp(candidate_weight_gradients);
        candidate_weight_gradients.TMult(dCv, input);
        candidate_weight_gradients += tmp;
    }
    if (candidate_state_weight_gradients.GetNoElements() > 0) {
        TMatrixT<Scalar_t> tmp(candidate_state_weight_gradients);
        candidate_state_weight_gradients.TMult(dCv, output_state);
        candidate_state_weight_gradients += tmp;
    }

    // For output gate
    if (output_weight_gradients.GetNoElements() > 0) {
        TMatrixT<Scalar_t> tmp(output_weight_gradients);
        output_weight_gradients.TMult(dOg, input);
        output_weight_gradients += tmp;
    }
    if (output_state_weight_gradients.GetNoElements() > 0) {
        TMatrixT<Scalar_t> tmp(output_state_weight_gradients);
        output_state_weight_gradients.TMult(dOg, output_state);
        output_state_weight_gradients += tmp;
    }

    // We've 4 bias vectors.
    if (input_bias_gradients.GetNoElements() > 0) {
      // This loops on state size.
        for (size_t j = 0; j < (size_t) dIg.GetNcols(); j++) {
            Scalar_t sum = 0.0;
            // This loops on batch size summing all gradient contributions in a batch.
            for (size_t i = 0; i < (size_t) dIg.GetNrows(); i++) {
                sum += dIg(i,j);
            }
        input_bias_gradients(j,0) += sum;
        }
    }
    // We'll follow similar pattern for other gates.
    if (forget_bias_gradients.GetNoElements() > 0) {
        for (size_t j = 0; j < (size_t) dFg.GetNcols(); j++) {
            Scalar_t sum = 0.0;
            // Loop over batchSize
            for (size_t i = 0; i < (size_t) dFg.GetNrows(); i++) {
                sum += dFg(i,j);
            }
        }
    }
    if (candidate_bias_gradients.GetNoElements() > 0) {
        for (size_t j = 0; j < (size_t) dCv.GetNcols(); j++) {
            Scalar_t sum = 0.0;
            // Loop over batchSize
            for (size_t i = 0; i < (size_t) dCv.GetNrows(); i++) {
                sum += dCv(i,j);
            }
        }
    }
    if (output_bias_gradients.GetNoElements() > 0) {
        for (size_t j = 0; j < (size_t) dOg.GetNcols(); j++) {
            Scalar_t sum = 0.0;
            // Loop over batchSize
            for (size_t i = 0; i < (size_t) dOg.GetNrows(); i++) {
                sum += dOg(i,j);
            }
        }
    }
}

} // namespace DNN
} // namespace TMVA
