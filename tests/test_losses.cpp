// loss checks: forward() must match the formula computed on the CPU,
// and backward() must match finite differences of forward()

#include "grad_check.h"
#include "loss/mse_loss.h"
#include "loss/cross_entropy_loss.h"
#include "loss/softmax_ce_loss.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

const int BATCH = 4;
const int CLASSES = 5;

// runs forward and returns the loss as a float
float loss_value(Loss* loss, Tensor* output, Tensor* target)
{
    Tensor* l = loss->forward(output, target);
    float value = to_host(l)[0];
    delete l;
    return value;
}

bool check_value(const char* name, float got, double want, double tol = 1e-5)
{
    double err = fabs(got - want) / fmax(1.0, fabs(want));
    bool ok = err <= tol;
    printf("  %-10s %-10s got %.6f want %.6f  %s\n", name, "value", got, want, ok ? "ok" : "FAIL");
    return ok;
}

// same idea as grad_check() for layers, but the loss is already a scalar so no weights are needed
bool loss_grad_check(const char* name, Loss* loss, Tensor* output, Tensor* target,
                     float h = 1e-3f, float tol = 1e-3f)
{
    // analytic (forward first: SoftmaxCE's backward uses probs saved by forward)
    delete loss->forward(output, target);
    Tensor* grad = loss->backward(output, target);
    std::vector<float> analytic = to_host(grad);
    delete grad;

    // numeric
    std::vector<float> values = to_host(output);
    std::vector<float> numeric(values.size());
    for (int i = 0; i < (int)values.size(); i++) {
        set_element(output, i, values[i] + h);
        double plus = loss_value(loss, output, target);

        set_element(output, i, values[i] - h);
        double minus = loss_value(loss, output, target);

        set_element(output, i, values[i]);
        numeric[i] = (float)((plus - minus) / (2.0 * h));
    }

    return report(name, "grad", analytic, numeric, tol);
}

// GPU tensor from host values
Tensor* make_tensor(const std::vector<float>& values, int rows, int cols)
{
    int shape[] = {rows, cols};
    Tensor* t = new Tensor(shape, 2, false);
    for (int i = 0; i < rows * cols; i++)
        t->data[i] = values[i];
    t->to_gpu();
    return t;
}

std::vector<float> random_one_hot(int rows, int cols)
{
    std::vector<float> v(rows * cols, 0.0f);
    for (int r = 0; r < rows; r++)
        v[r * cols + rand() % cols] = 1.0f;
    return v;
}

std::vector<float> softmax_rows(const std::vector<float>& x, int rows, int cols)
{
    std::vector<float> p(x.size());
    for (int r = 0; r < rows; r++) {
        float max_val = x[r * cols];
        for (int j = 1; j < cols; j++) max_val = fmaxf(max_val, x[r * cols + j]);

        double sum = 0.0;
        for (int j = 0; j < cols; j++) sum += exp(x[r * cols + j] - max_val);
        for (int j = 0; j < cols; j++) p[r * cols + j] = exp(x[r * cols + j] - max_val) / sum;
    }
    return p;
}

// -sum(target * log(p)) / rows
double cpu_cross_entropy(const std::vector<float>& p, const std::vector<float>& t, int rows)
{
    double sum = 0.0;
    for (size_t i = 0; i < p.size(); i++)
        sum -= t[i] * log(fmax(p[i], 1e-7));
    return sum / rows;
}

bool test_mse()
{
    int shape[] = {BATCH, CLASSES};
    Tensor* output = random_tensor(shape, 2);
    Tensor* target = random_tensor(shape, 2);
    std::vector<float> o = to_host(output), t = to_host(target);

    double want = 0.0;
    for (size_t i = 0; i < o.size(); i++)
        want += (o[i] - t[i]) * (o[i] - t[i]);
    want /= o.size();

    MSELoss loss;
    bool ok = check_value("mse", loss_value(&loss, output, target), want);
    ok = loss_grad_check("mse", &loss, output, target) && ok;

    delete output;
    delete target;
    return ok;
}

bool test_cross_entropy()
{
    // CrossEntropyLoss expects probabilities, so feed it softmax of random logits
    int shape[] = {BATCH, CLASSES};
    Tensor* logits = random_tensor(shape, 2);
    std::vector<float> p = softmax_rows(to_host(logits), BATCH, CLASSES);
    std::vector<float> t = random_one_hot(BATCH, CLASSES);
    delete logits;

    Tensor* probs  = make_tensor(p, BATCH, CLASSES);
    Tensor* target = make_tensor(t, BATCH, CLASSES);

    CrossEntropyLoss loss;
    bool ok = check_value("cross_ent", loss_value(&loss, probs, target), cpu_cross_entropy(p, t, BATCH));
    ok = loss_grad_check("cross_ent", &loss, probs, target) && ok;

    delete probs;
    delete target;
    return ok;
}

bool test_softmax_ce()
{
    int shape[] = {BATCH, CLASSES};
    Tensor* logits = random_tensor(shape, 2);
    std::vector<float> p = softmax_rows(to_host(logits), BATCH, CLASSES);
    std::vector<float> t = random_one_hot(BATCH, CLASSES);
    Tensor* target = make_tensor(t, BATCH, CLASSES);

    SoftmaxCELoss loss;
    bool ok = check_value("softmax_ce", loss_value(&loss, logits, target), cpu_cross_entropy(p, t, BATCH));
    ok = loss_grad_check("softmax_ce", &loss, logits, target) && ok;

    delete logits;
    delete target;
    return ok;
}

// SoftmaxCELoss allocates probs every forward; GPU memory must not grow over many calls
bool test_softmax_ce_no_leak()
{
    int shape[] = {64, 10};
    Tensor* logits = random_tensor(shape, 2);
    Tensor* target = make_tensor(random_one_hot(64, 10), 64, 10);
    SoftmaxCELoss loss;

    size_t free_before, free_after, total;
    delete loss.forward(logits, target);
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

    for (int i = 0; i < 10000; i++)
        delete loss.forward(logits, target);

    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    long leaked = (long)free_before - (long)free_after;

    bool ok = leaked < (1 << 20);  // allow 1 MB of allocator noise
    printf("  %-10s %-10s %ld bytes leaked over 10k forwards  %s\n", "softmax_ce", "memory", leaked, ok ? "ok" : "FAIL");

    delete logits;
    delete target;
    return ok;
}

int main()
{
    srand(0);
    bool ok = true;

    ok = test_mse()                && ok;
    ok = test_cross_entropy()      && ok;
    ok = test_softmax_ce()         && ok;
    ok = test_softmax_ce_no_leak() && ok;

    printf(ok ? "losses: all passed\n" : "losses: FAILED\n");
    return ok ? 0 : 1;
}
