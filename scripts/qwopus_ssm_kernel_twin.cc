// Standalone scalar twin of fst_ssm_scan_kernel.cc's recurrence (NO aie_api, NO
// transcendentals — the kernel is the pure delta rule; l2norm/scale/exp/sigmoid
// are upstream).  Validates the kernel's MATH against the numpy/transformers
// reference (scripts/qwopus_ssm_ref.py::gdn_delta_rule).  Reads stdin:
// HK*HV (S_in) + HK (qn) + HK (kn) + HV (v) + 1 (gdec) + 1 (beta) = 16770 fp32;
// writes stdout: HK*HV (S_out) + HV (y) = 16512 fp32.
#include <cstdio>
#include <vector>
int main() {
    constexpr int HK = 128, HV = 128;
    std::vector<float> in(HK*HV + HK + HK + HV + 1 + 1);
    for (auto &x : in) if (std::scanf("%f", &x) != 1) return 1;
    const float *Sin = in.data();
    const float *qn = Sin + HK*HV;
    const float *kn = qn + HK;
    const float *v = kn + HK;
    const float gdec = v[HV];
    const float beta = v[HV+1];
    std::vector<float> Sout(HK*HV), y(HV);

    std::vector<float> kvm(HV, 0.f);
    for (int i = 0; i < HK; ++i) {
        for (int j = 0; j < HV; ++j) {
            float s = Sin[i*HV+j] * gdec;
            Sout[i*HV+j] = s;
            kvm[j] += s * kn[i];
        }
    }
    std::vector<float> delta(HV);
    for (int j = 0; j < HV; ++j) delta[j] = (v[j] - kvm[j]) * beta;
    for (int j = 0; j < HV; ++j) y[j] = 0.f;
    for (int i = 0; i < HK; ++i) {
        for (int j = 0; j < HV; ++j) {
            float s = Sout[i*HV+j] + kn[i]*delta[j];
            Sout[i*HV+j] = s;
            y[j] += s * qn[i];
        }
    }
    for (float x : Sout) std::printf("%.9g\n", x);
    for (float x : y)     std::printf("%.9g\n", x);
    return 0;
}