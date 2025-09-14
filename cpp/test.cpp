#include <iostream>
#include <gtest/gtest.h>
#include "./test/tests.h"
#include "./src/sponge/sponge.h"

class HiResTimerListener : public ::testing::EmptyTestEventListener {
    std::chrono::high_resolution_clock::time_point start;
public:
    void OnTestStart(const ::testing::TestInfo&) override {
        start = std::chrono::high_resolution_clock::now();
    }
    void OnTestEnd(const ::testing::TestInfo& info) override {
        auto end = std::chrono::high_resolution_clock::now();
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << info.test_suite_name() << "." << info.name()
                  << " took " << nanos << " ns\n";
    }
};


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::UnitTest::GetInstance()->listeners().Append(new HiResTimerListener);
    return RUN_ALL_TESTS();
}
