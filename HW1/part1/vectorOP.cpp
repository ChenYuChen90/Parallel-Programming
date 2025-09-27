#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x, result;
  __pp_vec_float clamp = _pp_vset_float(9.999999f);
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_int y;
  __pp_mask maskAll, maskValue, maskExp, maskCount;

  for (int i = N; i < N+VECTOR_WIDTH; i++){
    values[i] = 0.0f;
    exponents[i] = 1;
  }
  
  for (int i = 0; i < N; i+=VECTOR_WIDTH){
    // All ones
    maskAll = _pp_init_ones();
    maskValue = _pp_init_ones(0);
    _pp_vset_float(result, 0.0f, maskExp);

    // Load vector of values, exp from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll);        // x = values[i];
    _pp_vload_int(y, exponents + i, maskAll);       // y = exponents[i];

    // Set mask according to predicate
    _pp_veq_int(maskExp, y, zero, maskAll);         // if (y == 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vset_float(result, 1.0f, maskExp);           // output[i] = 1.f;

    // Inverse maskIsNegative to generate "else" mask
    maskExp = _pp_mask_not(maskExp);                // } else {

    _pp_vmove_float(result, x, maskExp);            // result = x;
    _pp_vsub_int(y, y, one, maskExp);               // count = y - 1;
    _pp_vgt_int(maskCount, y, zero, maskAll);

    while (_pp_cntbits(maskCount) > 0){
      _pp_vmult_float(result, result, x, maskCount);// result *= x;

      _pp_vsub_int(y, y, one, maskCount);           // count--;
      _pp_vgt_int(maskCount, y, zero, maskAll);
    }

    _pp_vgt_float(maskValue, result, clamp, maskExp);// if (result > 9.999999f)

    _pp_vmove_float(result, clamp, maskValue);      // result = 9.999999f;

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);  // output[i] = result;
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_vec_float x, result;
  __pp_mask maskAll, maskResult;
  float sum = 0;
  maskAll = _pp_init_ones();
  maskResult = _pp_init_ones(1);
  _pp_vset_float(result, 0.f, maskAll);
  
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {  
    _pp_vload_float(x, values + i, maskAll);
    _pp_vadd_float(result, result, x, maskAll);
  }

  int Counter = VECTOR_WIDTH;
  while(Counter / 2 != 0){
    _pp_hadd_float(result, result);
    _pp_interleave_float(result, result);
    Counter /= 2;
  }
  
  _pp_vstore_float(&sum, result, maskResult);
  return sum;
}