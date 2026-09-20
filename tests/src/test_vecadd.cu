#include <gtest/gtest.h>
#include "vector.cuh"
#include <vector>
#include <cmath>

// 1. Тест базовой корректности сложения float
TEST(VectorAddTest, CorrectAdditionFloat) {
    constexpr size_t N = 1024;
    Vector<float> a(N);
    Vector<float> b(N);

    // Заполнение входных данных
    for (size_t i = 0; i < N; ++i) {
        a.view()[i] = static_cast<float>(i);
        b.view()[i] = static_cast<float>(i * 2);
    }

    // Выполнение сложения на GPU
    Vector<float> c = a + b;

    // Проверка результатов
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(c.view()[i], static_cast<float>(i * 3));
    }
}

// 2. Тест сложения большого вектора (перекрытие множества CUDA-блоков)
TEST(VectorAddTest, LargeVectorAddition) {
    constexpr size_t N = 1'000'000;
    Vector<int> a(N);
    Vector<int> b(N);

    for (size_t i = 0; i < N; ++i) {
        a.view()[i] = 10;
        b.view()[i] = 20;
    }

    Vector<int> c = a + b;

    for (size_t i = 0; i < 1000; ++i) { // Выборочная проверка для скорости
        EXPECT_EQ(c.view()[i], 30);
    }
    EXPECT_EQ(c.view()[N - 1], 30);
}

// 3. Тест на исключение при разной длине векторов
TEST(VectorAddTest, DimensionMismatchException) {
    Vector<float> a(100);
    Vector<float> b(200);

    EXPECT_THROW({
        Vector<float> c = a + b;
        }, std::invalid_argument);
}

// 4. Граничный случай: вектор из 1 элемента
TEST(VectorAddTest, SingleElement) {
    Vector<double> a(1);
    Vector<double> b(1);

    a.view()[0] = 3.14159;
    b.view()[0] = 2.71828;

    Vector<double> c = a + b;

    EXPECT_DOUBLE_EQ(c.view()[0], 3.14159 + 2.71828);
}