#include <cstddef>
#include <stdexcept>
#include <vector>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "vector.cuh"

namespace {

constexpr float kAbsTol = 1e-6f;

void fill_host(std::vector<float>& a, std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i) * 0.5f + 1.0f;
        b[i] = static_cast<float>(i) * 0.25f - 0.5f;
    }
}

}  // namespace

class VectorAddTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorAddTest, MatchesEigenVectorXf) {
    const std::size_t n = GetParam();

    std::vector<float> host_a(n);
    std::vector<float> host_b(n);
    fill_host(host_a, host_b);

    Eigen::VectorXf eigen_a = Eigen::Map<Eigen::VectorXf>(host_a.data(), static_cast<Eigen::Index>(n));
    Eigen::VectorXf eigen_b = Eigen::Map<Eigen::VectorXf>(host_b.data(), static_cast<Eigen::Index>(n));
    const Eigen::VectorXf eigen_sum = eigen_a + eigen_b;

    Vector<float> a(n);
    Vector<float> b(n);
    a.data().copy_from_host(host_a.data());
    b.data().copy_from_host(host_b.data());

    const Vector<float> c = a + b;

    std::vector<float> host_c(n);
    c.data().copy_to_host(host_c.data());

    const Eigen::Map<const Eigen::VectorXf> cuda_sum(
        host_c.data(), static_cast<Eigen::Index>(n));

    EXPECT_TRUE(cuda_sum.isApprox(eigen_sum, kAbsTol))
        << "Mismatch for n = " << n;
}

INSTANTIATE_TEST_SUITE_P(
    RequiredSizes,
    VectorAddTest,
    ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029));

TEST(VectorAddExtraTest, DimensionMismatchThrows) {
    const Vector<float> a(8);
    const Vector<float> b(16);
    EXPECT_THROW({ const Vector<float> c = a + b; }, std::invalid_argument);
}
