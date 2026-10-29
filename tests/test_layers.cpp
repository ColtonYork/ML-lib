// gradient checks for every layer: backward() must match finite differences of forward()

#include "grad_check.h"
#include "layers/linear.h"
#include "layers/relu.h"
#include "layers/softmax.h"
#include "cuda/cublas_ops.h"
#include <cstdio>
#include <cstdlib>

bool test_linear(CublasContext& ctx)
{
    bool ok = true;

    Linear lin(5, 3);
    lin.set_cublas(ctx);
    int shape[] = {4, 5};
    ok = grad_check("linear", &lin, shape, 2) && ok;

    // batch of 1 is a common edge case for the bias row-sum
    int single[] = {1, 5};
    ok = grad_check("linear b=1", &lin, single, 2) && ok;

    return ok;
}

bool test_relu()
{
    ReLU relu;
    int shape[] = {4, 5};
    return grad_check("relu", &relu, shape, 2);
}

bool test_softmax()
{
    bool ok = true;
    Softmax softmax;

    int shape[] = {4, 5};
    ok = grad_check("softmax", &softmax, shape, 2) && ok;

    // 3D input: softmax should run over the last dim only
    int shape_3d[] = {2, 3, 7};
    ok = grad_check("softmax 3d", &softmax, shape_3d, 3) && ok;

    // rows wider than the 256-thread block exercise the strided loop in the kernel
    int wide[] = {2, 300};
    ok = grad_check("softmax wd", &softmax, wide, 2) && ok;

    return ok;
}

int main()
{
    srand(0);
    CublasContext ctx;
    bool ok = true;

    ok = test_linear(ctx) && ok;
    ok = test_relu()      && ok;
    ok = test_softmax()   && ok;

    printf(ok ? "layers: all passed\n" : "layers: FAILED\n");
    return ok ? 0 : 1;
}
